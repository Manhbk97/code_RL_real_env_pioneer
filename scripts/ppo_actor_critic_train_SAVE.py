#!/usr/bin/env python
#import gfootball.env as football_env
import numpy as np
import rospy
import gym


import tensorflow as tf
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#tf.config.gpu_options.allow_growth = True
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = False
session = InteractiveSession(config=config)

#Disable GPU training
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from openai_ros.task_envs.turtlebot2 import turtlebot2_maze
from openai_ros.task_envs.turtlebot2 import turtlebot2_wall

import wandb
from wandb.keras import WandbCallback

wandb.init(name='PPO1', project="PPO Algorithm Testing")
#wandb.init(name='test_reward', project="PPO Algorithm Testing 2")

# Initialize your W&B project allowing it to sync with TensorBoard
wandb.init(name='PPO1', project="PPO Algorithm Testing", sync_tensorboard=True)
config = wandb.config


clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def ppo_loss_print(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        y_true = tf.Print(y_true, [y_true], 'y_true: ')
        y_pred = tf.Print(y_pred, [y_pred], 'y_pred: ')
        newpolicy_probs = y_pred
        # newpolicy_probs = y_true * y_pred
        newpolicy_probs = tf.Print(newpolicy_probs, [newpolicy_probs], 'new policy probs: ')

        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        ratio = tf.Print(ratio, [ratio], 'ratio: ')
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        actor_loss = tf.Print(actor_loss, [actor_loss], 'actor_loss: ')
        critic_loss = K.mean(K.square(rewards - values))
        critic_loss = tf.Print(critic_loss, [critic_loss], 'critic_loss: ')
        term_a = critic_discount * critic_loss
        term_a = tf.Print(term_a, [term_a], 'term_a: ')
        term_b_2 = K.log(newpolicy_probs + 1e-10)
        term_b_2 = tf.Print(term_b_2, [term_b_2], 'term_b_2: ')
        term_b = entropy_beta * K.mean(-(newpolicy_probs * term_b_2))
        term_b = tf.Print(term_b, [term_b], 'term_b: ')
        total_loss = term_a + actor_loss - term_b
        total_loss = tf.Print(total_loss, [total_loss], 'total_loss: ')
        return total_loss

    return loss


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss


def get_model_actor_image(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

    for layer in feature_extractor.layers:
        layer.trainable = False       #Freezing Layer               
    #Layers & models also feature a boolean attribute trainable. Its value can be changed. 
    #Setting layer.trainable to False moves all the layer's weights from trainable to non-trainable.

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    model.summary()
    return model


def get_model_actor_simple(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dense(128, activation='relu', name='fc3')(x)
    x = Dense(64, activation='relu', name='fc4')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    # model.summary()
    return model


def get_model_critic_image(input_dims):
    state_input = Input(shape=input_dims)

    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model


def get_model_critic_simple(input_dims):
    state_input = Input(shape=input_dims)

    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dense(128, activation='relu', name='fc3')(x)
    x = Dense(64, activation='relu', name='fc4')(x)
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    # model.summary()
    return model


def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        #state_input = K.expand_dims(state, 0)
        state_input = np.expand_dims(state, 0)
        # state_input = np.reshape(state,[1,state_dims])
        action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], batch_size=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        # limit += 1
        # if limit > 20:
        #     break
    return total_reward


def one_hot_encoding(probs):
    one_hot = np.zeros_like(probs)
    one_hot[:, np.argmax(probs, axis=1)] = 1
    return one_hot


#image_based = False
rospy.init_node('turtlebot2_maze_qlearn', anonymous=True, log_level=rospy.FATAL)
env_name = 'MyTurtleBot2Wall-v0'
env = gym.make(env_name)

# if image_based:
    
# else:
#     env = gym.make(env_name)

state = env.reset()
state_dims = env.observation_space.shape
n_actions = env.action_space.n

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

tensor_board = TensorBoard(log_dir='./logs')

# model_actor = get_model_actor_image(input_dims=state_dims, output_dims=n_actions)
# model_critic = get_model_critic_image(input_dims=state_dims)

# if image_based:

# else:
model_actor = get_model_actor_simple(input_dims=state_dims, output_dims=n_actions)
model_critic = get_model_critic_simple(input_dims=state_dims)

ppo_steps = 2048
target_reached = False
best_reward = 350
eps = 0
max_eps = 50000

while not target_reached and eps < max_eps:

    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = []
    actions_onehot = []
    state_input = None
    episode_reward, done = 0, False
    #env.render()
    env.reset()

    while not done:
        # state_input = K.expand_dims(state, 0)
        state_input = np.expand_dims(state, 0)
        # state_input = np.reshape(state,[1,state_dims])
        action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], batch_size=1)
        q_value = model_critic.predict([state_input], batch_size=1)
        action = np.random.choice(n_actions, p=action_dist[0, :])
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1



        observation, reward, done, info = env.step(action)
        print('ep: ' + str(eps) + ', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))
        mask = not done

        reward1 = np.reshape(reward, [1, 1])
        episode_reward += reward1[0][0]

        states.append(state)
        actions.append(action)
        actions_onehot.append(action_onehot)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)
        actions_probs.append(action_dist)

        state = observation
        # if done:
        #     env.reset()

    q_value = model_critic.predict(state_input, batch_size=1)
    values.append(q_value)
    returns, advantages = get_advantages(values, masks, rewards)
    # actor_loss = model_actor.fit(
    #     [states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
    #     [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=8,  
    #     callbacks=[tensor_board])

    # The WandbCallback logs metrics and some examples of the test data
    actor_loss = model_actor.fit(
        [states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
        [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=8, callbacks=[WandbCallback(),
        TensorBoard(log_dir=wandb.run.dir)])
    
    critic_loss = model_critic.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                   verbose=True, callbacks=[tensor_board])

    avg_reward = np.mean([test_reward() for _ in range(5)])
    print('total test reward=' + str(avg_reward))

    if avg_reward > best_reward:
        print('best reward=' + str(avg_reward))
        model_actor.save('model_actor_{}_{}.hdf5'.format(eps, avg_reward))
        model_critic.save('model_critic_{}_{}.hdf5'.format(eps, avg_reward))
        best_reward = avg_reward
    if best_reward > 400 or eps > max_eps:
        target_reached = True
    eps += 1
    print('number of episode: ' + str(eps))
    print('EP{} EpisodeReward={}'.format(eps, episode_reward))
    print('state dimension: ' + str(state_dims))
    print('action dimension: ' + str(n_actions))
    # print('state: ' + str(state) + str(np.shape(state)))
    # print('state input: ' + str(state_input))
    # print('state input1: ' + str(state_input1) + str(np.shape(state_input1)))
    # print('state input2: ' + str(state_input2) + str(np.shape(state_input2)))
    wandb.log({'Reward': episode_reward})
    wandb.log({'Total test reward': avg_reward})
    env.reset()

env.close()
