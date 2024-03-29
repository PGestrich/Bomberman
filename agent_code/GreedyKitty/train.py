from collections import namedtuple, deque

import pickle
from typing import List
from xml.dom import INDEX_SIZE_ERR

import events as e
from .callbacks import state_to_features, get_closest_coin_dist
import settings

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

import time

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
MOVING_EVENTS = np.array(['MOVED_UP','MOVED_RIGHT','MOVED_DOWN','MOVED_LEFT' ])
SUICIDE = "SUICIDE"
BACKWARDS = "BACKWARDS"
FORWARDS = "FORWARDS"
INTO_DANGER = "INTO_DANGER"
OUT_OF_DANGER = "OUT_OF_DANGER"
SURVIVED = "SURVIVED"
SWEET_SPOT = "SWEET_SPOT"
TOWARDS_COIN = "TOWARDS_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
CRATE_DESTROYED = "CRATE_DESTROYED"


#change in both callbacks & train!
dead_state = np.array([-100, -100, -100, -100, -100, -100]).reshape(1, -1)
new_prob = [10]*6
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


    

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    #y: array that saves rewards per action
    self.y = np.array([0]*6).reshape(1, -1)

    #X: array that saves state before action
    #self.X  = np.array([0, 0,  0, 0, 1, 20, 20, -1]).reshape(1, -1)
    self.X  = dead_state #np.array([0, 0,  0, 0, 0, 1]).reshape(1, -1)
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
        self.logger.debug(f'old_game_state is None')
        return None
    
    

    
    
    #self.logger.debug(f'model: {self.model}')
    # Idea: Add your own events to hand out rewards
    
    # state_to_features is defined in callbacks.py
    self.logger.debug(f"In game_events_occured")
    old_features = state_to_features(old_game_state, self, False)
    self.logger.debug(f"Old Features: {old_features}")
    new_features = state_to_features(new_game_state, self, False)
    self.logger.debug(f"New Features: {new_features}")

    
    # add events
    idx_action = ACTIONS.index(self_action)

    #escaping danger
    if idx_action != 5 and old_features[0][4] > 0 and  old_features[0][idx_action] == 0:
        events.append(OUT_OF_DANGER)
    if idx_action != 5 and old_features[0][idx_action] > 0:
        events.append(INTO_DANGER)
    if idx_action != 5 and old_features[0][idx_action] <= 0: #moving to safe places is good
        events.append(SWEET_SPOT)
    #waiting/bomb dropping on a dangerous field is bad
    if idx_action in [4,5] and old_features[0][4] == 1:
        events.append(INTO_DANGER)


    #move
    if idx_action in range(4) : 
        if abs(old_features[0][5] - idx_action) == 2:
            events.append(BACKWARDS)
        else:
            events.append(FORWARDS)
    
        #move toward a coin
    
    if 'COIN_COLLECTED' not in events and len(new_game_state['coins']) > 0 and 'COIN_FOUND' not in events:
        new_dist_coin = get_closest_coin_dist(new_game_state)
        old_dist_coin = get_closest_coin_dist(old_game_state)

        #print(new_dist_coin, old_dist_coin)
        #time.sleep(2)

        if new_dist_coin < old_dist_coin: # TODO
            events.append(TOWARDS_COIN)
        else:
            events.append(AWAY_FROM_COIN)
            

    #don't drop bombs if not allowed
    if idx_action == 5 and old_features[0][4] != -1:
        events.append(e.INVALID_ACTION)
    
    self.transitions.append(Transition(old_features, self_action, new_features,  reward_from_events(self, events)))

    self.y = augment_data(self, old_features, self_action, new_features, events)
    self.logger.debug(f'Encountered game event(s) {events} in step {new_game_state["step"]}\n')


                
    
    #update last action
    if idx_action in range(4):
        settings.move = idx_action
        self.logger.debug(f"Update action - now {settings.move} \n")
    if idx_action == 5:
        settings.move = 3 - settings.move
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
    self.logger.debug(f"In end_of_round")
    old_features = state_to_features(last_game_state, self, False)
    self.transitions.append(Transition(old_features, last_action, None, reward_from_events(self, events)))

    idx_action = ACTIONS.index(last_action)
    if old_features[0][idx_action] in [2,3]: #run into exploding fields is bad
        events.append(INTO_DANGER)
    
    #un-comment if using q_learn and not augment_data
    #idx_s = ((self.X == old_features).all(axis=1).nonzero())[0]
    #if len(idx_s) == 0:
    #    self.y = np.vstack((self.y, np.full((1,len(ACTIONS)), 1)))
    #    self.X = np.vstack((self.X, old_features))
    #    idx_s = [len(self.y)-1]
    
    # add events
    idx_action = ACTIONS.index(last_action)
    #self.y[idx_s, idx_action] = q_learn(self, old_features, last_action, dead_state, events, idx_s)
    self.y = augment_data(self, old_features, last_action, dead_state, events)

    #escaping danger
    if idx_action != 5 and old_features[0][4] > 0 and  old_features[0][idx_action] == 0:
        events.append(OUT_OF_DANGER)
    if idx_action != 5 and old_features[0][idx_action] > 0:
        events.append(INTO_DANGER)
    if idx_action != 5 and old_features[0][idx_action] <= 0: #moving to safe places is good
        events.append(SWEET_SPOT)
    
    #waiting/bomb dropping on a dangerous field is bad
    if idx_action in [4,5] and old_features[0][4] == 1:
        events.append(INTO_DANGER)
    

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
        e.INVALID_ACTION: -1000,
        #e.BOMB_EXPLODED: 100,
        e.BOMB_DROPPED: 15,
        #e.KILLED_SELF: -20,
        INTO_DANGER: -50,
        OUT_OF_DANGER: 30,
        BACKWARDS: -10,
        FORWARDS: 10, 
        SWEET_SPOT: 10, # idea: the custom event is bad
        TOWARDS_COIN: 50,
        AWAY_FROM_COIN: -50,
    }
    reward_sum = 0

    while e.BOMB_DROPPED in events and e.CRATE_DESTROYED not in events: # the agent gets points for bombs only when it bombs a box
        events.remove(e.BOMB_DROPPED)
    
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def q_learn(self, old_state, self_action: str, new_state, events: List[str], idx_s)-> int:
    """
    Update data using Q-learning.
    """
    # define parameters
    GAMMA = 0.5
    ALPHA = 0.5

    
    
    idx_action = ACTIONS.index(self_action)
    #self.logger.debug(f'action {self_action}, idx {idx_action}')

    reward = reward_from_events(self, events)
    
    
    q_old = self.y[idx_s, idx_action]

    # in the beginning
    # [[-100. -100. -100. -100. -100. -100.]
    # [   2.    2.    0.    2.   -1. -100.]]
    # [ 2.  2.  0.  2. -1.  3.]


    #get index of Q(s_t+1):
    idx_new =  ((self.X == new_state).all(axis=1).nonzero())[0]
    #not present
    if len(idx_new) == 0:
        self.y = np.vstack((self.y, new_prob))
        self.X = np.vstack((self.X, new_state))
        idx_new = [len(self.y)-1]
    
    new_pred = self.y[idx_new]
    q_max = np.amax(new_pred)

    q_new = q_old + ALPHA*(reward + GAMMA*q_max - q_old)
    q_new = np.maximum(q_new, 0)
    self.logger.debug(f'change {q_old} to  {q_new} with reward {reward}, q_max = {q_max} since new_state= {new_pred}')

    return q_new


def augment_data(self, state, action: str, new_state, events: List[str]):
    """
    Use symmetries in the Game to create more training data
    """
    
    if new_state is None:
        return self.y

    self.logger.debug(f'In augment data')

    #test data
    #state = np.array([[3,1,5,7, 0]])
    #new_state = np.array([[4,2,6,8, 0]])
    
    adjacent = state[0][0:4]
    new_adjacent = new_state[0][0:4]
    movement = state[0][5]
    

    
    rot_action = ACTIONS.index(action)

    rotate = [3, 0, 1,2]
    self.logger.debug(f'adjacent {adjacent}, new_adjacent {new_adjacent}, action {rot_action} / {action}')
    tested = []

    #test without function:
    #rot_state = np.append(adjacent, state[0][4:6])
    #idx_s =  ((self.X == rot_state).all(axis=1).nonzero())[0]
    #if len(idx_s) == 0:
    #    self.y = np.vstack((self.y, np.full((1,len(ACTIONS)), 1)))
    #    self.X = np.vstack((self.X, rot_state))
    #    idx_s = [len(self.y)-1]
    #self.y[idx_s, rot_action] = q_learn(self, rot_state, ACTIONS[rot_action], np.append(new_adjacent, [new_state[0][4], movement]), events, idx_s)
    #idx_action = ACTIONS.index(action)
    
    
    #return self.y

    #rotate
    for i in range(4):
        adjacent = adjacent[rotate]
        new_adjacent = new_adjacent[rotate]

        if movement != -100:
                movement = (movement + 1) % 4
        
        #rotate action if necessary
        if rot_action in range(4):
            rot_action = (rot_action + 1) % 4
            
            events = update_event(events)
        #don't double waiting + bomb actions
        elif np.any([np.array_equal(np.append(adjacent, rot_action), ar) for ar in tested]):
            continue
        
        rot_state = np.append(adjacent, [state[0][4], movement])
        #learn
        self.logger.debug(f'Rot Features: {rot_state}, Action: {ACTIONS[rot_action]}')
        self.logger.debug(f'New Features: {np.append(new_adjacent, new_state[0][4])}')
        tested.append(rot_state)

        idx_s = ((self.X == rot_state).all(axis=1).nonzero())[0]
        if len(idx_s) == 0:
            self.y = np.vstack((self.y, new_prob))
            self.X = np.vstack((self.X, rot_state))
            idx_s = [len(self.y)-1]
        
        new_movement = movement
        if rot_action in range(4):
            new_movement = rot_action
        self.y[idx_s, rot_action] = q_learn(self, rot_state, ACTIONS[rot_action], np.append(new_adjacent, [new_state[0][4], new_movement]), events, idx_s)
        
        self.logger.debug(f' New Probablities: {self.y[idx_s]}')
        
    
    


    #mirrored states
    rot_action = ACTIONS.index(action)
    adjacent = state[0][[2,1,0,3]]
    new_adjacent = new_state[0][[2,1,0,3]]
    if rot_action in [0, 2]:
        rot_action = (rot_action + 2) % 4
        if movement != -100:
            movement = (movement + 2) %4
        events = update_event(update_event(events))
    
    
        
    #else process mirrored states too
    for i in range(4):
        adjacent = adjacent[rotate]
        new_adjacent = new_adjacent[rotate]
        if movement != -100:
                movement = (movement + 1)%4

        if rot_action in range(4):
            rot_action = (rot_action + 1) % 4
            events = update_event(events)
        rot_state = np.append(adjacent, [state[0][4], movement])
        
        
        #return self if state is mirror-symmetric
        #self.logger.debug(f'tested : {tested}')
        #self.logger.debug(f'to test {rot_state}')
        #self.logger.debug(f'{[np.array_equal(np.append(adjacent, [rot_action, movement]), ar) for ar in tested]}')
        if np.any([np.array_equal(rot_state, ar) for ar in tested]):
            self.logger.debug(f' Mirror symmetric')
            return self.y

        
        self.logger.debug(f'Rot2 Features: {rot_state}, Action: {ACTIONS[rot_action]}')
        self.logger.debug(f'New2 Features: {np.append(new_adjacent, new_state[0][4])}')
        tested.append(rot_state)
        
        idx_s = ((self.X == rot_state).all(axis=1).nonzero())[0]

        if len(idx_s) == 0:
            self.y = np.vstack((self.y, new_prob))
            self.X = np.vstack((self.X, rot_state))
            idx_s = [len(self.y)-1]
        

        new_movement = movement
        if rot_action in range(4):
            new_movement = rot_action
        self.y[idx_s, rot_action] = q_learn(self,rot_state, ACTIONS[rot_action], np.append(new_adjacent, [new_state[0][4], new_movement]), events, idx_s)
        self.logger.debug(f' New Probablities: {self.y[idx_s]}')

    return self.y


def update_event(events):
    for id in range(len(events)):
        event = events[id]
        idx = ((MOVING_EVENTS == event).nonzero())[0]
        if idx.size > 0 and idx[0] in range(4):
            events[id] = MOVING_EVENTS[(idx + 1)%4][0]
        
    return events