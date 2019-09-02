#Import libraries used in the project
import cv2
import vizdoom as vzd
from argparse import ArgumentParser
from collections import deque
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt

#Vizdoom config file used
DEFAULT_CONFIG = "defend_the_center.cfg"
#Hardcoded resolution to be used for state/image size 
RESOLUTION = (42, 42)
#Learning rate of our model
LEARNING_RATE = 0.0001
#Hardcoded memory size (keep it small)
REPLAY_SIZE = 10000
#Batch size used for training our network
BATCH_SIZE = 32
#Probability of taking a random action
EPSILON = 0.05
#Discount factor
GAMMA = 0.99
#Minimum number of experiences before we start training
SAMPLES_TILL_TRAIN = 1000
#How often model is updated - in terms of agent steps
UPDATE_RATE = 4
#How often we should save the model - in terms of agent steps
SAVE_MODEL_EVERY_STEPS = 10000
#Number of frames in frame stack - number of successive frames provided to agent
FRAME_STACK_SIZE = 4
#Hardcoded model path - used for both loading and saving
MODEL_PATH = "model.h5"
#Game rate 
GAME_RATE = 10
#Number of episodes for training
MAX_EPISODES = 3000

class ReplayMemory:
    """Simple implementation of replay memory for DQN

    Stores experiences (s, a, r, s') in circulating
    buffer
    """
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        # Original state
        self.s1 = np.zeros((capacity, ) + state_shape, dtype=np.uint8)
        # Successor state
        self.s2 = np.zeros((capacity, ) + state_shape, dtype=np.uint8)
        # Action taken
        self.a = np.zeros(capacity, dtype=np.int)
        # Reward gained
        self.r = np.zeros(capacity, dtype=np.float)
        # If s2 was terminal or not
        self.t = np.zeros(capacity, dtype=np.uint8)

        # Current index in circulating buffer,
        # and total number of items in the memory
        self.index = 0
        self.num_total = 0

    def add_experience(self, s1, a, r, s2, t):
        # Turn states into uint8 to save some space
        self.s1[self.index] = (s1 * 255).astype(np.uint8)
        self.a[self.index] = a
        self.r[self.index] = r
        self.s2[self.index] = (s2 * 255).astype(np.uint8)
        self.t[self.index] = t

        self.index += 1
        self.num_total = max(self.index, self.num_total)
        # Return to beginning if we reach end of the buffer
        if self.index == self.capacity:
            self.index = 0

    def sample_batch(self, batch_size):
        """Return batch of batch_size of random experiences

        Returns experiences in order s1, a, r, s2, t.
        States are already normalized
        """
        # Here's a small chance same experience will occur twice
        indexes = np.random.randint(0, self.num_total, size=(batch_size,))
        # Normalize images to [0, 1] (networks really don't like big numbers).
        # They are stored in buffers as uint8 to save space.
        return [
            self.s1[indexes] / 255.0,
            self.a[indexes],
            self.r[indexes],
            self.s2[indexes] / 255.0,
            self.t[indexes],
        ]
def build_models(input_shape, action_size):
    """Build Keras models for predicting Q-values

    And returns Model
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=[6, 6], strides=[4, 4], activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=[3, 3], strides=[3, 3], activation='relu'))
    model.add(Conv2D(64, kernel_size=[3, 3], strides=[1, 1], activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_dim=512, activation='relu'))
    model.add(Dense(output_dim=action_size, 
                    activation=None,
                    kernel_initializer=keras.initializers.Zeros(),
                    bias_initializer=keras.initializers.Zeros())
            )

    model.compile(loss='mse',optimizer=Adam(lr=LEARNING_RATE))

    return model
def update_model(model, replay_memory, batch_size=BATCH_SIZE):
    """Run single update step on model and return loss"""

    s1, a, r, s2, t = replay_memory.sample_batch(batch_size)
    s2_values = np.max(model.predict(s2), axis=1)

    target_values = r + GAMMA * s2_values * (1 - t)

    # Get the Q-values for first state. This will work as
    # the base for target output (the "y" in prediction task)
    s1_values = model.predict(s1)
    # Update Q-values of the actions we took with the new
    # target_values we just calculated.
    
    # for i in range(batch_size):
    #     s1_values[i, a[i]] = target_values[i]
    s1_values[np.arange(batch_size), a] = target_values

    # Finally, run the update through network
    loss = model.train_on_batch(s1, s1_values)
    return loss
def get_action(s1, model, num_actions):
    """Return action to be taken in s1 according to Q-values from model or random action in case of exploration"""
    
    # Get Q values using same model 
    q_values = model.predict(s1[None])[0]

    action = None
    #Based on EPSILON value given perform Exploitation or Exploration
    if random.random() < EPSILON:
        action = random.randint(0, num_actions - 1)
    else:
        action = np.argmax(q_values)
    return action, q_values
def preprocess_state(state, stacker):
    """Handle stacking frames, and return state with multiple consecutive frames"""
    
    #Transpose and reshape the given state to have same size of RESOLUTION
    state = state.transpose([1, 2, 0])
    state = cv2.resize(state, RESOLUTION)
    
    #Convert the given rgb image to grayscale
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    
    # Normalize to [0,1]
    state = state.astype(np.float) / 255.0 
    
    # Add channel dimension to match network input_shape
    state = state[...,None]
    
    # Add to the stacker
    stacker.append(state)
    # Create proper state to be used by the network
    stacked_state = np.concatenate(stacker, axis=2)
    return stacked_state
def create_actions(listy, n, max_ones=1):
    """
    returns a list of all combinations of 0's and 1's of length n with max number of 1's=max_ones
    """
    if sum(listy) >= max_ones:
        if len(listy) < n:
            listy += [0]*(n-len(listy))
        return listy
    if len(listy) >= n:
        return listy
    return put_it_in_a_single_list([create_actions(listy + [0], n, max_ones), create_actions(listy + [1], n, max_ones)])
def put_it_in_a_single_list(listy):
    result = []
    def recurse(listy):
        for obj in listy:
            if type(obj) != list:
                result.append(listy)
                break
            else:
                recurse(obj)
    recurse(listy)
    return result
def main(args):
    #Setup the game and initialize it
    game = vzd.DoomGame()
    game.load_config(DEFAULT_CONFIG)
    game.set_sound_enabled(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(False)
    game.init()
    #Set action size and state shape which will be used as input shape to our network
    action_size = game.get_available_buttons_size()
    state_shape = RESOLUTION + (FRAME_STACK_SIZE,)
    state_stacker = deque(maxlen=FRAME_STACK_SIZE)

    #Initialize replay memory (if training)
    replay_memory = None
    if not args.evaluate:
        replay_memory = ReplayMemory(REPLAY_SIZE, state_shape)
    
    step_cnt = 0
    previous_episode_rewards = deque(maxlen=100)
    
    model = None
    #Construct new model or load existing model 
    if not args.evaluate:
        model = build_models(
            state_shape,
            action_size
        )
    else:
        model = keras.models.load_model(MODEL_PATH)
    #Total episodic rewards for final plotting purpose
    tot_epi_rewards = []
    for step_cnt in range(MAX_EPISODES):
        game.new_episode()
        game_state = game.get_state()        
        #Prepare our actions in an array format
        actions = create_actions([], action_size, 1)
        episode_reward = 0
        losses = []
        q_values_sum = 0
        q_values_num = 0
        #Clear existing values in state_stacker and append new values of zeros
        state_stacker.clear()
        for i in range(FRAME_STACK_SIZE):
            state_stacker.append(np.zeros(RESOLUTION + (1,)))
        #Get current state
        s1 = game.get_state().screen_buffer
        #Preprocess state
        s1 = preprocess_state(s1,state_stacker)
        while not game.is_episode_finished():
            step_cnt += 1           
            #Choose whether to do exploration and exploitation and get action based accordingly
            a,q_values = get_action(s1, model, action_size)
            q_values_sum += q_values.sum()
            q_values_num += len(q_values)
            #Make action and get reward
            r = game.make_action(actions[a], GAME_RATE)
            s2 = game.get_state()
            #Check if state is not None before performing preprocessing
            if s2 is not None:
                s2 = s2.screen_buffer
                s2 = preprocess_state(s2,state_stacker)
                done = game.is_episode_finished()
                episode_reward += r
                replay_memory.add_experience(s1,a,r,s2,done)
            #Check if we should do updates or saving model
            if (step_cnt % UPDATE_RATE) == 0:
                if replay_memory.num_total > SAMPLES_TILL_TRAIN:
                    losses.append(update_model(model, replay_memory))
            if (step_cnt % SAVE_MODEL_EVERY_STEPS) == 0:
                        model.save(MODEL_PATH)
            
            s1 = s2
        
        
        #Calculate average loss and average Q value
        if len(losses) != 0:
            avrg_loss = sum(losses) / len(losses)
        else:
            avrg_loss = 0
        if q_values_num != 0:
            avrg_q = q_values_sum / q_values_num
        else:
            avrg_q = 0
        
        
        previous_episode_rewards.append(episode_reward)
        avrg_reward = sum(previous_episode_rewards) / len(previous_episode_rewards)   
        tot_epi_rewards.append(avrg_reward)
        s = "Episode reward: %.1f\tAvrg reward: %.3f\tSteps: %d\tLoss: %.5f\tQ: %.5f" % (
                episode_reward, avrg_reward, step_cnt, avrg_loss, avrg_q
            )
        print(s)
    
    #Close the game and plot the graph of training rewards 
    game.close()
    print(len(tot_epi_rewards))
    plt.plot(tot_epi_rewards)
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.title("training episodic rewards for 'defend_the_center'")
    plt.show()
       
if __name__ == "__main__":
    parser = ArgumentParser("Train DQN on VizDoom.")
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="Evaluate the model instead of training")
    args = parser.parse_args()

    main(args)
