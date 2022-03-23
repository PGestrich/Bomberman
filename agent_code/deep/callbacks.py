import os
import pickle
import random

import numpy as np
import tensorflow as tf


from tensorflow.keras import models, layers, utils, backend as K


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

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    

    

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #
    else:
        self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
            #model =  pickle.load(file)
        self.model = models.load_model("my-saved-model.keras")
    


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    features = state_to_features(game_state)
    #print('features: ', features, features.shape)
    pred = self.model.predict(features)[0]

    self.logger.debug(f"features: {features}")
    self.logger.debug(f"prediction: {pred}")

    random_prob = .2
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.logger.debug(f"Chosen action: {action}\n")
        return action

    action = 0
    if np.sum(pred) == 0:
        action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])
    else:
        action = ACTIONS[np.argmax(pred)]

    
    #action = ACTIONS[int(pred[0,0])]
    self.logger.debug(f"Chosen action: {action}\n")

    return action
    
    """
    # todo Exploration vs exploitation 
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)
    """


def state_to_features(game_state: dict) -> np.array:
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
        return None
    
    field = game_state['field']
    position = game_state['self'][3]
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = game_state['others']

    top, top_pos = 0, (position[0] , position[1] - 1)
    right, right_pos = 1, (position[0] + 1, position[1])
    bottom, bottom_pos = 2, (position[0] , position[1] + 1)
    left, left_pos = 3, (position[0]  - 1, position[1])
    current = 4
    adjacent = np.array([field[top_pos],
                        field[right_pos],
                        field[bottom_pos],
                        field[left_pos]])
    neighbours = [top_pos, right_pos, bottom_pos, left_pos]

    for agent in others:
        agent_pos = agent[3]
        for i in range(4):
            if agent_pos == neighbours[i]:
                adjacent[i] = -1

    for bomb in bombs:
        bomb_pos = bomb[0]
        for i in range(4):
            if bomb_pos == neighbours[i]:
                adjacent[i] = -1
    
    coin = BFS_coin(field, position, coins)
    
    adjacent = np.append(adjacent, coin)
    return adjacent.reshape(1, -1)

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

def BFS_coin(field, position, coins):    
    """
    Use Breadth-First-Search to find opponents and coins
    """
    #self.logger.debug(f' in BFS_opponent ')
    x_occupied, y_occupied = np.where(field != 0)
    explored = [(x_occupied[i], y_occupied[i]) for i in range(len(x_occupied))]

    amount = 0
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[position]]

    o_top =(position[0] , position[1] - 1)
    o_right= (position[0] + 1, position[1])
    o_bottom= (position[0] , position[1] + 1)
    o_left = (position[0]  - 1, position[1])

    res = -100
    found_opponent = False
    
    if coins == []:
        return -100
     
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        amount += 1
        path = queue.pop(0)
        node = path[-1]

        if node in explored:
            continue

        
        top =(node[0] , node[1] - 1)
        right= (node[0] + 1, node[1])
        bottom= (node[0] , node[1] + 1)
        left = (node[0]  - 1, node[1])
        
        neighbours = [top, right, bottom, left]
        np.random.shuffle(neighbours)

        # Loop to iterate over the
        # neighbours of the node
        for neighbour in neighbours:
            if neighbour in explored:
                continue
            
            #check if found coin - return first step
            if neighbour in coins:

                path.append(neighbour)
                if path[1] == o_top:
                    res = 0
                elif path[1] == o_right:
                    res = 1
                elif path[1] == o_bottom:
                    res = 2
                else:
                    res = 3
                
                return res
            
            #else continue sesarch
            new_path = path.copy()    
            new_path.append(neighbour)
            queue.append(new_path)

            
            
        explored.append(node)
 
    
    #if opponent found, but no coins: return oponent
    #if nothing found, return original value (-100)
    return res
