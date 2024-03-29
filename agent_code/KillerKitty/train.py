from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
import settings

import numpy as np

from sklearn.tree import DecisionTreeRegressor

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
TOWARDS_OPPONENT = "TOWARDS OPPONENT"
AVOID_OPPONENT = "AVOID OPPONENT"

TOWARDS_CRATE = "TOWARDS CRATE"
AVOID_CRATE = "AVOID CRATE"

TOWARDS_COIN = "TOWARDS COIN"
AVOID_COIN = "AVOID COIN"

SMART_BOMB = "SMART BOMB"
OK_BOMB = "OK BOMB"
DUMB_BOMB = "DUMB BOMB"



#change in both callbacks & train!
dead_state = np.array([-100]*8).reshape(1, -1)
new_prob = [100]*6
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
        #old_game_state = None_state
        self.logger.debug(f'old_game_state is None')
        return None
    
    

    
    
    #self.logger.debug(f'model: {self.model}')
    # Idea: Add your own events to hand out rewards
    
    # state_to_features is defined in callbacks.py
    self.logger.debug(f"In game_events_occured")
    old_features = state_to_features(old_game_state, self, True)
    self.logger.debug(f"Old Features: {old_features}")
    new_features = state_to_features(new_game_state, self, False)
    #self.logger.debug(f"New Features: {new_features}")
    

    
    # add events:
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


    
    #targeted movement
    if idx_action in range(4) :
        #moving towards opponents is good 
        if old_features[0][5] == idx_action:
            events.append(TOWARDS_COIN)
        elif old_features[0][5] != -100:
            events.append(AVOID_COIN)
        #crates not as imporant, but still good
        else:
            if old_features[0][6] == idx_action:
                events.append(TOWARDS_CRATE)
            if 3 - old_features[0][6] == idx_action:
                events.append(AVOID_CRATE)
            
            if old_features[0][7] == idx_action:
                events.append(TOWARDS_OPPONENT)
            if 3 - old_features[0][7] == idx_action:
                events.append(AVOID_OPPONENT)
            
    
    
    
    if idx_action == 5:
        #don't drop bombs if not allowed
        if old_features[0][4] != -1:
            events.append(e.INVALID_ACTION)
        #encourage dropping bonbs next to crates or opponents
        if old_features[0][5]!= -100 and old_features[0][old_features[0][5]] == 2:
            events.append(SMART_BOMB)
        #allow bomb dropping if not harmful
        elif old_features[0][6]!= -100 and old_features[0][old_features[0][6]] == 2:
            if old_features[0][5] != -100:
                events.append(OK_BOMB)
            else:
                events.append(SMART_BOMB)
        else:
            events.append(DUMB_BOMB)
    
    self.transitions.append(Transition(old_features, self_action, new_features,  reward_from_events(self, events)))
    #self.y[idx_s, idx_action] = q_learn(self, old_features, self_action, new_features, events, idx_s)

    self.y = augment_data(self, old_features, self_action, new_features, events)
    self.logger.debug(f'Encountered game event(s) {events} in step {new_game_state["step"]}\n')

    if old_game_state['step'] % 10 == 0:
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


    # add events:
    idx_action = ACTIONS.index(last_action)

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


    
    #targeted movement
    if idx_action in range(4) :
        #moving towards opponents is good 
        if old_features[0][5] == idx_action:
            events.append(TOWARDS_COIN)
        elif old_features[0][5] != -100:
            events.append(AVOID_COIN)
        #crates not as imporant, but still good
        else:
            if old_features[0][6] == idx_action:
                events.append(TOWARDS_CRATE)
            if 3 - old_features[0][6] == idx_action:
                events.append(AVOID_CRATE)
            
            if old_features[0][7] == idx_action:
                events.append(TOWARDS_OPPONENT)
            if 3 - old_features[0][7] == idx_action:
                events.append(AVOID_OPPONENT)
            
    
    
    
    if idx_action == 5:
        #don't drop bombs if not allowed
        if old_features[0][4] != -1:
            events.append(e.INVALID_ACTION)
        #encourage dropping bonbs next to crates or opponents
        if old_features[0][5]!= -100 and old_features[0][old_features[0][5]] == 2:
            events.append(SMART_BOMB)
        #allow bomb dropping if not harmful
        elif old_features[0][6]!= -100 and old_features[0][old_features[0][6]] == 2:
            if old_features[0][5] != -100:
                events.append(OK_BOMB)
            else:
                events.append(SMART_BOMB)
        else:
            events.append(DUMB_BOMB)



    
    idx_action = ACTIONS.index(last_action)
    self.y = augment_data(self, old_features, last_action, dead_state, events)
    
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
        #e.BOMB_DROPPED: -5,
        #e.KILLED_SELF: -20,
        INTO_DANGER: -50,
        OUT_OF_DANGER: 30,
        TOWARDS_COIN: 30,
        AVOID_COIN: -30, 

        TOWARDS_CRATE: 15,
        AVOID_CRATE: -5, 

        TOWARDS_OPPONENT: 15,
        AVOID_OPPONENT: -5, 

        #SUICIDE : -100,
        SWEET_SPOT: 10, 
        SMART_BOMB: 150,
        OK_BOMB: 5,
        DUMB_BOMB: -25
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


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
    
    new_pred = self.y[idx_new]
    q_max = np.amax(new_pred)

    #compute new value
    q_new = q_old + ALPHA*(reward + GAMMA*q_max - q_old)
    q_new = np.maximum(q_new, 0)
    if log:
        self.logger.debug(f'change {q_old} to  {q_new} with reward {reward}, q_max = {q_max} since new_state= {new_pred}')

    return q_new


def augment_data(self, state, action: str, new_state, events: List[str]):
    """
    Use symmetries in the Game to create more training data
    """

    #once had error that new_state was none. Seems to happen rarely, so well just ignore training data with that input
    if new_state is None:
        return self.y

    self.logger.debug(f'In augment data')
    reward = reward_from_events(self, events)

    #get original features ready to rotate
    adjacent = state[0][0:4]
    new_adjacent = new_state[0][0:4]
    coin = state[0][5]
    crate = state[0][6]
    opp = state[0][7]
    new_coin = new_state[0][5]
    new_crate = new_state[0][6]
    new_opp = new_state[0][7]
    

    #rotate parameters + already tested
    rot_action = ACTIONS.index(action)
    rotate = [3, 0, 1,2]
    tested = []

    #rotate
    for i in range(4):
        adjacent = adjacent[rotate]
        new_adjacent = new_adjacent[rotate]

        #rotate last two features
        if coin != -100:
            coin = (coin + 1) % 4
        if crate != -100:
            crate = (crate + 1) % 4
        if opp != -100:
            opp = (opp + 1) % 4
        if new_coin != -100:
            new_coin = (new_coin + 1) % 4
        if new_crate != -100:
            new_crate = (new_crate + 1) % 4
        if new_opp != -100:
            new_opp = (new_opp + 1) % 4
        
        #rotate action if necessary
        if rot_action in range(4):
            rot_action = (rot_action + 1) % 4
            
            events = update_event(events)
        
        #don't double existing test cases
        elif np.any([np.array_equal(np.append(adjacent, rot_action), ar) for ar in tested]):
            continue
        
        rot_state = np.append(adjacent, [state[0][4], coin, crate, opp])
        new_rot_state = np.append(new_adjacent, [new_state[0][4], new_coin, new_crate, new_opp])
        tested.append(rot_state)

        idx_s = ((self.X == rot_state).all(axis=1).nonzero())[0]
        if len(idx_s) == 0:
            self.y = np.vstack((self.y, new_prob))
            self.X = np.vstack((self.X, rot_state))
            idx_s = [len(self.y)-1]
        
        
        
        #update rotated data    
        self.y[idx_s, rot_action] = q_learn(self, rot_state, ACTIONS[rot_action], new_rot_state, reward, idx_s)
        
    self.logger.debug(f' New Probablities: {self.y[idx_s]}')
        
    
    


    #mirrored states
    rot_action = ACTIONS.index(action)
    adjacent = state[0][[2,1,0,3]]
    new_adjacent = new_state[0][[2,1,0,3]]
    coin = state[0][5]
    crate = state[0][6]
    opp = state[0][7]
    new_coin = new_state[0][5]
    new_crate = new_state[0][6]
    new_opp = new_state[0][7]

    #mirror everything if necessary
    if rot_action in [0, 2]:
        rot_action = (rot_action + 2) % 4
        if coin != -100:
            coin = (coin + 2) %4
        if crate != -100:
            crate = (crate + 2) % 4
        if opp != -100:
           opp = (opp + 2) % 4
        if new_coin != -100:
            new_coin = (new_coin + 2) % 4
        if new_crate != -100:
            new_crate = (new_crate + 2) % 4
        if new_opp != -100:
           new_opp = (new_opp + 2) % 4
        events = update_event(update_event(events))
    
    
        
    # process mirrored states (aka rotating 2.0)
    for i in range(4):
        adjacent = adjacent[rotate]
        new_adjacent = new_adjacent[rotate]
        if coin != -100:
            coin = (coin + 1)%4
        if crate != -100:
            crate = (crate + 1) % 4
        if opp != -100:
           opp = (opp + 1) % 4
        if new_coin != -100:
            new_coin = (new_coin + 1) % 4
        if new_crate != -100:
            new_crate = (new_crate + 1) % 4
        if new_opp != -100:
           new_opp = (new_opp + 2) % 4

        if rot_action in range(4):
            rot_action = (rot_action + 1) % 4
            events = update_event(events)
        rot_state = np.append(adjacent, [state[0][4], coin, crate, opp])
        new_rot_state = np.append(new_adjacent, [new_state[0][4], new_coin, new_crate, new_opp])
        
        
        if np.any([np.array_equal(rot_state, ar) for ar in tested]):
            self.logger.debug(f' Mirror symmetric')
            return self.y

        tested.append(rot_state)
        
        idx_s = ((self.X == rot_state).all(axis=1).nonzero())[0]
        if len(idx_s) == 0:
            self.y = np.vstack((self.y, new_prob))
            self.X = np.vstack((self.X, rot_state))
            idx_s = [len(self.y)-1]
        
        self.y[idx_s, rot_action] = q_learn(self,rot_state, ACTIONS[rot_action], new_rot_state, reward, idx_s)

    return self.y


def update_event(events):
    for id in range(len(events)):
        event = events[id]
        idx = ((MOVING_EVENTS == event).nonzero())[0]
        if idx.size > 0 and idx[0] in range(4):
            events[id] = MOVING_EVENTS[(idx + 1)%4][0]
        
    return events
