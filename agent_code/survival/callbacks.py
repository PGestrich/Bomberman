from calendar import c
import os
import pickle
import random

import settings

import numpy as np
from sklearn.tree import DecisionTreeRegressor

move = -100

#change in both callbacks & train!
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
dead_state = np.array([-100, -100, -100, -100, -100, -100]).reshape(1, -1)
new_prob = [100]*6



def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = DecisionTreeRegressor(random_state=0)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.\n")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    #self.logger.debug("Querying model for action.")
    features = state_to_features(game_state, self)
    self.logger.debug(f'Adjacent:  {features}')

    
    prediction = self.model.predict(features)[0]
    self.logger.debug(f'Prediction:  {prediction}')  

    if np.sum(prediction) == 0:
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.logger.debug(f'Action:  {action}\n')
        return action
    
    action = np.random.choice(ACTIONS, p=prediction/np.sum(prediction))
    self.logger.debug(f'Action:  {action}\n')

    if not self.train:
        idx_action = ACTIONS.index(action)
        global move
        if idx_action in range(4):
            move = idx_action
        if idx_action == 6:
            move = 3 - idx_action
    return action


def state_to_features(game_state: dict, self) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return  np.array([-100, -100, -100, -100, -100, -100, -100]).reshape(1, -1)
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    name, score, bomb, position = game_state['self']
    others = game_state['others']
    user_input = game_state['user_input']
    self.logger.debug(f'position: {position}, bombs: {bombs}')

    #meaning of numbers
    exploding = 3
    occupied = 2
    countdown = 1
    
    #top, right, bottom, left, current
    top, top_pos = 0, (position[0] , position[1] - 1)
    right, right_pos = 1, (position[0] + 1, position[1])
    bottom, bottom_pos = 2, (position[0] , position[1] + 1)
    left, left_pos = 3, (position[0]  - 1, position[1])
    current = 4
    adjacent = np.array([occupied*abs(field[top_pos]),
                        occupied*abs(field[right_pos]),
                        occupied*abs(field[bottom_pos ]),
                        occupied*abs(field[left_pos]),
                        - bomb])
    
    for agent in others:
        agent_pos = agent[3]
        if agent_pos == top_pos:
            adjacent[top] = occupied
        if agent_pos == right_pos:
            adjacent[right] = occupied
        if agent_pos == bottom_pos:
            adjacent[bottom] = occupied
        if agent_pos == left_pos:
            adjacent[left] = occupied
                        
    #return adjacent.reshape(1, -1)
    explosion = np.array([explosion_map[top_pos ],
                        explosion_map[right_pos],
                        explosion_map[bottom_pos ],
                        explosion_map[left_pos],
                        explosion_map[position]
                        ])
    info = adjacent + exploding*explosion
    
    
    current_bomb = [0,0]
    for i in list(bombs):
        b_pos = i[0]
        current_bomb[0] = b_pos[0] - position[0]
        current_bomb[1] = b_pos[1] - position[1]
        
        #bombs block fields
        if b_pos == top_pos:
            info[top] = occupied
        elif b_pos == right_pos:
            info[right] = occupied
        elif b_pos == bottom_pos:
            info[bottom] = occupied
        elif b_pos == left_pos:
            info[left] = occupied
            
        

        # update danger of fields
        if (i[1] == 0):
            countdown = exploding
        if np.linalg.norm(current_bomb) <= settings.BOMB_POWER:
            #self.logger.debug(f'current bomb: {current_bomb} - i[1] + 1: {i[1] + 1}')
            #self.logger.debug(f'Bomb Position: {b_pos} - own position: {position}')
            
            #check for bombs: same line (e.g. 3 steps up)
            if current_bomb[0] == 0:
                if info[current] != exploding and np.linalg.norm(current_bomb) < settings.BOMB_POWER:
                    info[current] = np.maximum(countdown, info[current])
                    #self.logger.debug(f'bomb same position')

                if current_bomb[1] <= 0 and info[top] != exploding:
                    info[top] = np.maximum(countdown, info[top])
                    #self.logger.debug(f'bomb up')
                if current_bomb[1] >= 0 and info[bottom] != exploding:
                    info[bottom] = np.maximum(countdown, info[bottom])
                    #self.logger.debug(f'bomb down')
                
            if current_bomb[1] == 0:
                if info[current] != exploding and np.linalg.norm(current_bomb) < settings.BOMB_POWER:
                    info[current] = np.maximum(countdown, info[current])
                    #self.logger.debug(f'bomb same position')
                    
                if current_bomb[0] <= 0 and info[left] != exploding:
                    info[left] = np.maximum(countdown, info[left])
                    #self.logger.debug(f'bomb left')
                if current_bomb[0] >= 0 and info[right] != exploding:
                    info[right] = np.maximum(countdown, info[right])
                    #self.logger.debug(f'bomb right')
            
            #check for bombs: crossing (e.g. one step up, three to the side)
            if current_bomb[1] == -1 and info[top] != exploding:
                info[top] = np.maximum(countdown, info[top])
                #self.logger.debug(f'bomb crossing up')
            if current_bomb[1] == 1 and info[bottom] != exploding:
                info[bottom] = np.maximum(countdown, info[bottom])
                #self.logger.debug(f'bomb crossing down')

            if current_bomb[0] == -1 and info[left] != exploding:
                info[left] = np.maximum(countdown, info[left])
                #self.logger.debug(f'bomb crossing left')
            if current_bomb[0] == 1 and info[right] != exploding:
                info[right] = np.maximum(countdown, info[right])
                #self.logger.debug(f'bomb crossing right')
                
    movement = 0
    if self.train:
        movement = settings.move
    else:
        global move
        movement = move
    self.logger.debug(f'adjacent : {adjacent}, explosion : {explosion}')
    self.logger.debug(f'info : {info}, move: {movement}')
    info = np.append(info, movement)

    return info.reshape(1, -1)
    
    #get position of nearest bomb
    nearest_bomb = [20, 20]
    boom = -1
    #self.logger.debug(f'Position : {position}')
    for i in list(bombs):
        b_pos = i[0]
        #self.logger.debug(f'Bomb Position: {b_pos} - nearest: {nearest_bomb}')
        if np.linalg.norm(np.subtract(b_pos, position)) < np.linalg.norm(nearest_bomb):
            nearest_bomb[0] = b_pos[0] - position[0]
            nearest_bomb[1] = b_pos[1] - position[1]
            #self.logger.debug(f'closer!')
            boom = i[1]
    #check explosion map:
    if explosion_map[position[0], position[1] - 1] > 0:
        #self.logger.debug(f'1 boom!')
        nearest_bomb = [0, -1]
        boom = 0
    if explosion_map[position[0] + 1, position[1]] > 0:
        #self.logger.debug(f'2 boom!')
        nearest_bomb = [1, 0]
        boom = 0
    if explosion_map[position[0], position[1] + 1] > 0:
        #self.logger.debug(f'3 boom!')
        nearest_bomb = [0, 1]
        boom = 0
    if explosion_map[position[0] - 1, position[1] ] > 0:
        #self.logger.debug(f'4 boom!')
        nearest_bomb = [-1, 0]
        boom = 0

    # For example, you could construct several channels of equal shape, ...
    channels = adjacent.tolist() + [bomb] + nearest_bomb + [boom]
    #channels.append(bomb)
    #channels.append(list(position))
    #channels.append(list(bombs))
    
    

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    #print(stacked_channels)
    # and return them as a vector
    return stacked_channels.reshape(1, -1)
