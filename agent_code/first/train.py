from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

from sklearn.tree import DecisionTreeRegressor

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
dead_state = [0]*6
SUICIDE = "SUICIDE"
None_state = np.array([-100, -100, -100, -100, -100, -100]).reshape(1, -1)



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    #y: array that saves rewards per action
    self.y = np.array([10, 10, 10, 10, 5, 0]).reshape(1, -1)

    #X: array that saves state before action
    #self.X  = np.array([0, 0,  0, 0, 1, 20, 20, -1]).reshape(1, -1)
    self.X  = None_state #np.array([0, 0,  0, 0, 0, 1]).reshape(1, -1)
    self.model.fit(self.X, self.y)

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    Wour knowledge of the (new) game state.

    This is *one* of the places whehen this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and yre you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state is None:
        #old_game_state = None_state
        self.logger.debug(f'old_game_state is None')
        return None
    
    self.logger.debug(f'Encountered game event(s) {events} in step {new_game_state["step"]}')
    
    #self.logger.debug(f'model: {self.model}')
    # Idea: Add your own events to hand out rewards
    
    # state_to_features is defined in callbacks.py
    old_features = state_to_features(old_game_state, self)
    new_features = state_to_features(new_game_state, self)
    self.logger.debug(f"Features: {old_features}")
    
    #Index: find if state was already present in dataset
    idx_s = ((self.X == old_features).all(axis=1).nonzero())[0]
    


    #not present
    if len(idx_s) == 0:
        self.y = np.vstack((self.y, np.full((1,len(ACTIONS)), 1)))
        self.X = np.vstack((self.X, old_features))
        idx_s = [len(self.y)-1]
    
    #self.logger.debug(f'idx_s: {idx_s}')    
    #self.logger.debug(f'update {self.y[idx_s, idx_action]} to {(self.y[idx_s, idx_action] + reward)/2}')    
    idx_action = ACTIONS.index(self_action)
    self.transitions.append(Transition(old_features, self_action, new_features,  reward_from_events(self, events)))
    self.y[idx_s, idx_action] = q_learn(self, old_features, self_action, new_features, events, idx_s)
    #self.y[idx_s, idx_action] = (self.y[idx_s, idx_action] + reward)/2
    self.model.fit(self.X, np.nan_to_num(self.y))

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
    old_features = state_to_features(last_game_state, self)
    self.transitions.append(Transition(old_features, last_action, None, reward_from_events(self, events)))

    #learn from last step
    #Index: find if state was already present in dataset
    idx_s = ((self.X == old_features).all(axis=1).nonzero())[0]
    #not present
    if len(idx_s) == 0:
        self.y = np.vstack((self.y, np.full((1,len(ACTIONS)), 1)))
        self.X = np.vstack((self.X, old_features))
        idx_s = [len(self.y)-1]
    
    #self.logger.debug(f'idx_s: {idx_s}')    
    #self.logger.debug(f'update {self.y[idx_s, idx_action]} to {(self.y[idx_s, idx_action] + reward)/2}')    
    idx_action = ACTIONS.index(last_action)
    self.y[idx_s, idx_action] = q_learn(self, old_features, last_action, dead_state, events, idx_s)
    #self.y[idx_s, idx_action] = (self.y[idx_s, idx_action] + reward)/2
    
    #unlearn suicidal behaviour
    if 'KILLED_SELF' in events:
        print_state = last_game_state['self'][3]
        explosion_map = last_game_state['explosion_map']
        bombs = last_game_state['bombs']
        #self.logger.debug(f'Position: {print_state}')
        #self.logger.debug(f'Bombs: {bombs}')
        #self.logger.debug(f'Explosion map: {explosion_map}')
        
        
        for t in self.transitions:
            self.logger.debug(f'transition: {t.action}')
            #discourage bomb dropping - for some reason it kills itself due to waiting?
            if t.action in ['BOMB', 'WAIT']:
                idx_s = ((self.X == t.state).all(axis=1).nonzero())[0]
                q_learn(self, t.state, t.action, dead_state, ['SUICIDE'], idx_s)
            #Discourage other behaviours
            else:
                idx_s = ((self.X == t.state).all(axis=1).nonzero())[0]
                q_learn(self, t.state, t.action, dead_state, ['KILLED_SELF'], idx_s)
    self.model.fit(self.X, np.nan_to_num(self.y))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -100,
        e.WAITED: 5,
        e.MOVED_UP: 10,
        e.MOVED_DOWN: 10,
        e.MOVED_RIGHT: 10,
        e.MOVED_LEFT: 10,
        e.BOMB_DROPPED: 1,
        e.GOT_KILLED: 0,
        e.KILLED_SELF: 0,
        SUICIDE: -10  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def q_learn(self, old_state, self_action: str, new_state, events: List[str], idx_s)-> int:
    """
    Calculate new reward
    """

    GAMMA = 0.5
    ALPHA = 0.8

    
    
    idx_action = ACTIONS.index(self_action)
    self.logger.debug(f'action {self_action}, idx {idx_action}')

    reward = reward_from_events(self, events)
    
    
    q_old = self.y[idx_s, idx_action]

    #get index of Q(s_t+1):
    idx_new =  ((self.X == new_state).all(axis=1).nonzero())[0]
    #not present
    if len(idx_new) == 0:
        self.y = np.vstack((self.y, np.full((1,len(ACTIONS)), 1)))
        self.X = np.vstack((self.X, new_state))
        idx_new = [len(self.y)-1]
    
    q_max = np.amax(self.y[idx_new])

    q_new = q_old + ALPHA*(reward + GAMMA*q_max - q_old)
    q_new = np.maximum(q_new, 0)
    self.logger.debug(f'change {q_old} to  {q_new} with reward {reward}\n')

    return q_new

