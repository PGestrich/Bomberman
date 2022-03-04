import os
import pickle
import random

import numpy as np
from sklearn.tree import DecisionTreeRegressor


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
        if self.train:
            idx_s = ((self.X == features).all(axis=1).nonzero())[0]
            self.y[idx_s] = [10, 10, 10, 10, 5, 5]
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.logger.debug(f'Action:  {action}\n')
        return action
    
    action = np.random.choice(ACTIONS, p=prediction/np.sum(prediction))
    self.logger.debug(f'Action:  {action}\n')
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
        return  np.array([-100, -100, -100, -100, -100, -100]).reshape(1, -1)
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    name, score, bomb, position = game_state['self']
    others = game_state['others']
    user_input = game_state['user_input']
    
    #top, right, bottom, left, current
    adjacent = np.array([field[position[0]    , position[1] - 1],
                        field[position[0] + 1, position[1]],
                        field[position[0]    , position[1] + 1],
                        field[position[0] - 1, position[1]],
                        0])
                        
    #return adjacent.reshape(1, -1)
    adjacent = abs(adjacent)

    if explosion_map[position[0], position[1] - 1] > 0:
        self.logger.debug(f'1 boom!')
        nearest_bomb = [0, -1]
        boom = 0
    if explosion_map[position[0] + 1, position[1]] > 0:
        self.logger.debug(f'2 boom!')
        nearest_bomb = [1, 0]
        boom = 0
    if explosion_map[position[0], position[1] + 1] > 0:
        self.logger.debug(f'3 boom!')
        nearest_bomb = [0, 1]
        boom = 0
    if explosion_map[position[0] - 1, position[1] ] > 0:
        self.logger.debug(f'4 boom!')
        nearest_bomb = [-1, 0]
        boom = 0
    explosion = np.array([explosion_map[position[0]    , position[1] - 1],
                        explosion_map[position[0] + 1, position[1]],
                        explosion_map[position[0]    , position[1] + 1],
                        explosion_map[position[0] - 1, position[1]],
                        explosion_map[position[0]    , position[1]]
                        ])
    info = np.maximum(adjacent, explosion)
    
    
    current_bomb = [0,0]
    for i in list(bombs):
        b_pos = i[0]
        current_bomb[0] = b_pos[0] - position[0]
        current_bomb[1] = b_pos[1] - position[1]
        #self.logger.debug(f'Bomb Position: {b_pos} - nearest: {nearest_bomb}')
        countdown = -i[1] - 1
        if np.linalg.norm(current_bomb) < 6:
            self.logger.debug(f'current bomb: {current_bomb} - i[1]: {i[1]}')
            if current_bomb[1] == 0:
                if current_bomb[0] < 0:
                    info[0] = countdown
                    self.logger.debug(f'bomb up')
                if current_bomb[0] > 0:
                    info[2] = countdown
                    self.logger.debug(f'bomb down')
                if current_bomb[0] == 0:
                    info[4] = countdown
                    self.logger.debug(f'bomb same position')
            if current_bomb[0] == 0:
                if current_bomb[1] < 0:
                    info[3] = countdown
                    self.logger.debug(f'bomb left')
                if current_bomb[1] > 0:
                    info[1] = countdown
                    self.logger.debug(f'bomb right')
    
    self.logger.debug(f'adjacent : {adjacent}, explosion : {explosion}, info : {info}')
    return np.append(info, bomb).reshape(1, -1)
    
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
