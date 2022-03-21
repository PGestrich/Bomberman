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
new_prob = [100]*6

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
    outputs = layers.Dense(name="output", units=1, activation='softmax')(h2)
    self.model = models.Model(inputs=inputs, outputs=outputs)


    #training
    def R2(y, y_hat):
        ss_res =  K.sum(K.square(y - y_hat)) 
        ss_tot = K.sum(K.square(y - K.mean(y))) 
        return ( 1 - ss_res/(ss_tot + K.epsilon()) )


    self.model.compile(optimizer='adam', loss='mean_absolute_error',  metrics=[tf.keras.metrics.MeanSquaredError()])


    self.X = np.array([[0,0,0,0, -100 ], [0,0,0,0,0]])
    self.y =  np.array([0, 0])
    self.rewards = np.array([0, 1])

    #print(self.X, self.X.shape)
    #print(self.y, self.y.shape)
    self.model.fit(self.X, self.y, verbose = 0)



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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    move_events = [e.MOVED_UP, e.MOVED_RIGHT, e.MOVED_LEFT, e.MOVED_DOWN]
    
    for e in events:
        if e in move_events:
            events.append(MOVED)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    features = state_to_features(old_game_state)
    if features is None:
        return None


    idx_s = ((self.X == features).all(axis=1).nonzero())[0]
    if len(idx_s) == 0:
        self.y = np.append(self.y, random.randint(0,4) )
        self.X = np.vstack((self.X, features))
        self.rewards = np.append(self.rewards, -1)
        idx_s = [len(self.y)-1]

    reward = reward_from_events(self, events)

    self.logger.debug(f'current reward: {self.y[idx_s]} new reward {self.rewards[idx_s]}')
    self.logger.debug(f'therefore {self.rewards[idx_s] < reward}')
    #update model
    idx_action = ACTIONS.index(self_action)   
    
    if self.rewards[idx_s] < reward:
        y_old = self.y[idx_s]
        self.y[idx_s] = idx_action
        self.rewards[idx_s] = reward
        self.logger.debug(f'updated for features {features} from {y_old} to {idx_action} with reward {reward}')

    self.model.fit(self.X, self.y, verbose = 0)



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
    self.model.save('my-saved-model.keras')


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION : -100,
        e.COIN_COLLECTED: 10,
       # e.KILLED_OPPONENT: 50,
        e.BOMB_DROPPED: -50,
        MOVED: 5  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}\n")
    return reward_sum
