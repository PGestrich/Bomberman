from collections import namedtuple, deque

import pickle
from typing import List

import events as e
import numpy as np
from .callbacks import state_to_features

import tensorflow as tf
import random



from tensorflow.keras import models, layers, utils, backend as K

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
MOVED = "MOVED"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
new_prob = [100, 100, 100, 100, 00, 0]

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    D = len(ACTIONS)
    n_features = 5
    inputs = layers.Input(name="input", shape=(n_features,))

    #layers
    h1 = layers.Dense(name="h1", units=n_features, activation='relu')(inputs)
    #h1 = layers.Dropout(name="drop1", rate=0.1)(h1) #overfitting should not be a problem
    h2 = layers.Dense(name="h2", units=n_features, activation='relu')(h1)

    #output
    outputs = layers.Dense(name="output", units=len(ACTIONS), activation='softmax')(h2)
    self.model = models.Model(inputs=inputs, outputs=outputs)
    


    #training
    def R2(y, y_hat):
        ss_res =  K.sum(K.square(y - y_hat)) 
        ss_tot = K.sum(K.square(y - K.mean(y))) 
        return ( 1 - ss_res/(ss_tot + K.epsilon()) )

    cce = tf.keras.losses.SparseCategoricalCrossentropy()

    self.model.compile(optimizer='adam', loss=cce,  metrics=['sparse_categorical_accuracy'])
    
    #self.training_model = models.Model(inputs=inputs, outputs=outputs)
    #self.training_model = self.model.compile(optimizer='adam', loss=cce,  metrics=[tf.keras.metrics.MeanSquaredError()])


    self.X = np.array([[0,0,0,0, -100 ], [0,0,0,0,0]])

    
    self.y =  np.array([[5,5,5,5,0,0], [15,5,5,5,0,0]])
    #print(self.y.shape)
    #print(self.y.shape)
    self.y_max = np.empty(self.y.shape[0])
    
    for idx, v in enumerate(self.y):
        maximum = np.where(v == v.max())[0]
        self.y_max[idx] = random.choice(maximum)

    
    #self.y_unit = np.array([v/np.linalg.norm(v) for v in self.y])
    #self.y_max = np.array([random.choice(np.argmax(v)) for v in self.y])
    
    #print(self.y_unit)
    #print(self.X, self.X.shape)
    #print(self.y, self.y.shape)
    self.model.fit(self.X, self.y_max, verbose = 0)


    


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    move_events = ["MOVED_UP", "MOVED_RIGHT", "MOVED_LEFT", "MOVED_DOWN"]

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    
    
    
    for e in events:
        if e in move_events:
            events.append(MOVED)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    features = state_to_features(old_game_state)
    if features is None:
        return None
    
    new_features = state_to_features(new_game_state)
    if new_features is None:
        return None


   # idx_s = ((self.X == features).all(axis=1).nonzero())[0]
    
    """if len(idx_s) == 0:
        self.y = np.append(self.y, random.randint(0,4) )
        self.X = np.vstack((self.X, features))
        self.rewards = np.append(self.rewards, -1)
        idx_s = [len(self.y)-1]

    reward = reward_from_events(self, events)

    self.logger.debug(f'current action: {self.y[idx_s]}/reward: {self.rewards[idx_s]} new reward {reward}')
    self.logger.debug(f'therefore {self.rewards[idx_s] < reward}')
    #update model
    idx_action = ACTIONS.index(self_action)   
    
    if self.rewards[idx_s] < reward:
        y_old = self.y[idx_s]
        self.y[idx_s] = idx_action
        self.rewards[idx_s] = reward
        self.logger.debug(f'updated for features {features} from {y_old} to {idx_action} with reward {reward}')
    """
    idx_action = ACTIONS.index(self_action)
    idx_s = ((self.X == features).all(axis=1).nonzero())[0]
    if len(idx_s) == 0:
        self.y = np.vstack((self.y, new_prob))
        self.X = np.vstack((self.X, features))
        idx_s = [len(self.y)-1]
        self.y_max = np.append(self.y_max, 0)
    
    reward = reward_from_events(self, events)
    self.y[idx_s, idx_action] = q_learn(self, features, self_action, new_features, reward, idx_s, True)


    maximum = np.where(self.y[idx_s]== self.y[idx_s].max())[0]
    self.y_max[idx_s] = random.choice(maximum)

def q_learn(self, old_state, self_action: str, new_state, reward, idx_s, log = False)-> int:
    """
    Update data using Q-learn
    """
    #define parameters
    GAMMA = 0.2
    ALPHA = 0.9

    
    
    idx_action = ACTIONS.index(self_action)
    
    #get all imporant values
    q_old = self.y[idx_s, idx_action]

    idx_new =  ((self.X == new_state).all(axis=1).nonzero())[0]
    if len(idx_new) == 0:
        self.y = np.vstack((self.y, new_prob))
        self.X = np.vstack((self.X, new_state))
        idx_new = [len(self.y)-1]
        self.y_max = np.append(self.y_max, 0)
    
    new_pred = self.y[idx_new]
    q_max = np.amax(new_pred)

    #compute new value
    q_new = q_old + ALPHA*(reward + GAMMA*q_max - q_old)
    q_new = np.maximum(q_new, 0)
    if log:
        self.logger.debug(f'change {q_old} to  {q_new} with reward {reward}, q_max = {q_max} since new_state= {new_pred}')

    return np.maximum(q_new,0)

    



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    #with open("my-saved-model.pt", "wb") as file:
    #self.y_unit = np.array([v/np.linalg.norm(v) for v in self.y])
    if random.randint(0,100) == 5:
        self.model.fit(self.X, self.y_max)
    else:
         self.model.fit(self.X, self.y_max, verbose = 0)
    self.model.save('my-saved-model.keras')


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION : -100,
        e.COIN_COLLECTED: 100,
       # e.KILLED_OPPONENT: 50,
        e.WAITED: -50,
        e.BOMB_DROPPED: -50,
        MOVED: 5  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}\n")
    return reward_sum
