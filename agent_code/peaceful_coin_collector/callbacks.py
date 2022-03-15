from cmath import inf
import os
import pickle
import random

import numpy as np
from sklearn.tree import DecisionTreeRegressor
#from sklearn.linear_model import Lasso


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
dead_state = np.array([-100, -100, -100, -100, -100, -100]).reshape(1, -1)

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
        self.is_fit = False # to fix a problem
        self.model = DecisionTreeRegressor(random_state=0)
        #self.model = Lasso(random_state=0)
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
    #if self.train and random.random() < random_prob or not self.is_fit:
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
        if self.train and features[0][0] != -100:
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

    #print(game_state)
    
    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    name, score, bomb, position = game_state['self'] # bomb: if bomb is possible
    others = game_state['others'] # all opponents
    user_input = game_state['user_input']
    self.logger.debug(f'position: {position}')
    
    # 0: no special property
    # 1: dangerous (will explode soon)
    # 2: is blocked
    # 3: will kill the agent

    #top, right, bottom, left, current
    top, top_pos = 0, (position[0] , position[1] - 1)
    right, right_pos = 1, (position[0] + 1, position[1])
    bottom, bottom_pos = 2, (position[0] , position[1] + 1)
    left, left_pos = 3, (position[0]  - 1, position[1])
    current = 4
    adjacent = np.array([2*abs(field[top_pos]),
                        2*abs(field[right_pos]),
                        2*abs(field[bottom_pos ]),
                        2*abs(field[left_pos]),
                        - bomb]) # describes in which direction around the position you can go e.g. top, bottom, right, left - [0, 0, -1, 0]
    
    for agent in others: # to describe other agents
        agent_pos = agent[3]
        if agent_pos == top_pos:
            adjacent[top] = 2 # to mark that you have agents on your way: blocked
        if agent_pos == right_pos:
            adjacent[right] = 2
        if agent_pos == bottom_pos:
            adjacent[bottom] = 2
        if agent_pos == left_pos:
            adjacent[left] = 2
                        
    #return adjacent.reshape(1, -1)
    explosion = np.array([explosion_map[top_pos ],
                        explosion_map[right_pos],
                        explosion_map[bottom_pos ],
                        explosion_map[left_pos],
                        explosion_map[position]  # how many more steps an explosion will be present for "position" tile, 0 for no explosion
                        ])
                        
    
    info = adjacent + 3*explosion # info about explosion e.g. [0, 0, 0, 1, 0] where 1 is bomb: bomb on the left
    
    # peaceful coin collector doesn't drop any bomb
    """
    current_bomb = [0,0]
    for i in list(bombs):
        b_pos = i[0] # bomb possibility for the first tile
        current_bomb[0] = b_pos[0] - position[0]
        current_bomb[1] = b_pos[1] - position[1]
        #self.logger.debug(f'Bomb Position: {b_pos} - nearest: {nearest_bomb}')
        
        exploding = 3
        countdown = 1
        if (i[1] == 0):
            countdown = exploding
        if np.linalg.norm(current_bomb) < 5:
            #self.logger.debug(f'current bomb: {current_bomb} - i[1] + 1: {i[1] + 1}')
            #self.logger.debug(f'Bomb Position: {b_pos} - own position: {position}')
            
            #check for bombs: same line (e.g. 3 steps up)
            if current_bomb[0] == 0:
                if info[current] != exploding:
                    info[current] = countdown # = no free tiles!
                    #self.logger.debug(f'bomb same position')

                if current_bomb[1] < 0 and info[top] != exploding:
                    info[top] = countdown
                    #self.logger.debug(f'bomb up')
                if current_bomb[1] > 0 and info[bottom] != exploding:
                    info[bottom] = countdown
                    #self.logger.debug(f'bomb down')
                
            if current_bomb[1] == 0:
                if info[current] != exploding:
                    info[current] = countdown
                    #self.logger.debug(f'bomb same position')
                    
                if current_bomb[0] < 0 and info[left] != exploding:
                    info[left] = countdown
                    #self.logger.debug(f'bomb left')
                if current_bomb[0] > 0 and info[right] != exploding:
                    info[right] = countdown
                    #self.logger.debug(f'bomb right')
            
            #check for bombs: crossing (e.g. one step up, three to the side)
            if current_bomb[1] == -1 and info[top] != exploding:
                info[top] = countdown
                #self.logger.debug(f'bomb crossing up')
            if current_bomb[1] == 1 and info[bottom] != exploding:
                info[bottom] = countdown
                #self.logger.debug(f'bomb crossing down')

            if current_bomb[0] == -1 and info[left] != exploding:
                info[left] = countdown
                #self.logger.debug(f'bomb crossing left')
            if current_bomb[0] == 1 and info[right] != exploding:
                info[right] = countdown
                #self.logger.debug(f'bomb crossing right')

    """

    self.logger.debug(f'adjacent : {adjacent}, explosion : {explosion}, info : {info}')
    return info.reshape(1, -1)



def search_coin(game_state):
    """A function that finds the shortest path to the next coin (alternatively crate) and returns the next step to the coin"""
    # to be deleted later:
    # field: all the information about the field(-1: stone walls, 0: free tiles, 1: crates)
    # coins: coordinates of all coins
    # position: where the agent is
    # adjacent: describes in which direction from the position you can go e.g. top, bottom, right, left - [0, 0, -1, 0] # you can't go to the right

    graph = get_free_neighbors(game_state) # a dict  {(1, 1): [(1, 0), (2, 1), (0, 1)], ...}
    field = game_state['field']
    coins = game_state['coins']
    name, score, bomb, position = game_state['self']

    move = None
    shortest_path = inf

    for coin in coins:
        path, path_length = BFS_SP(graph, position, coin) # find shortest path to coin # e.g. ([(0, 1), (1, 1), (2, 1)], 2)
        
        if path_length < shortest_path:
            shortest_path = path_length
            move = path[0] # ? TODO
    return move

"""
move = -1
shortest_path = inf
for coin in coins: 
  find_shortest_path_to_coin
  if path_length < shortest_path:
      shotest_path = path_length
      move = first_step
return move
"""


def get_free_neighbors(game_state):
    """create a graph as a dict for possible movements from each position"""

    field = game_state['field']

    direction = [(0,-1), (1,0), (0,1), (-1,0)] 

    x_0, y_0 = np.where(field == 0)

    free_tiles = [(x_0[i], y_0[i]) for i in range(len(x_0))] # coordinates of free tiles


    targets = []
    for coord in free_tiles: # for each coordinate...
        pos_movement = [tuple(map(sum, zip(coord, n))) for n in direction]

        targets.append([x for x in pos_movement if x in free_tiles]) # if the coordinate is in free tiles

    graph = dict(zip(free_tiles, targets))

    return graph # {(1, 1): [(1, 0), (2, 1), (0, 1)], ...}


def BFS_SP(graph, start, goal):
    """from https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/"""
    explored = []
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[start]]
     
    # If the desired node is
    # reached
    if start == goal:
        print("Same Node")
        return 0
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
         
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    #print("Shortest path = ", *new_path)
                    return [*new_path], len(new_path)-1 # return shortest path and the length
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    #print("connecting path doesn't exist")
    #return