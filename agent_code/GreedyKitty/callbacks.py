from calendar import c
import os
import pickle
import random
from cmath import inf

import settings

import numpy as np
from sklearn.neighbors import RadiusNeighborsRegressor

move = -100

#change in both callbacks & train!
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
dead_state = np.array([-100, -100, -100, -100, -100, -100]).reshape(1, -1)
new_prob = [10]*6



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
        self.model = RadiusNeighborsRegressor(random_state=0)

    else:
        self.logger.info("Loading model from saved state.")
        global move
        move = -100
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
    


    self.logger.debug(f"In act")
    features = state_to_features(game_state, self, True)
    self.logger.debug(f'Adjacent:  {features}')

    
    prediction = self.model.predict(features)[0]
    self.logger.debug(f'Prediction:  {prediction}')  

    if np.sum(prediction) == 0:
        action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])
    else:
        action = np.random.choice(ACTIONS, p=prediction/np.sum(prediction))
    idx_action = ACTIONS.index(action)
    self.logger.debug(f'Action:  {action} - idx: {idx_action}\n')

    if not self.train:
        global move
        if idx_action in range(4):
            move = idx_action
            self.logger.debug(f'changed move to {move}')
        if idx_action == 5:
            move = 3 - move
            self.logger.debug(f'changed move to {move}\n')
    return action


def state_to_features(game_state: dict, self, log:bool) -> np.array:
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

    # commented out variables are not used
    # round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    # coins = game_state['coins']
    _, _, bomb, position = game_state['self']
    others = game_state['others']
    # user_input = game_state['user_input']
    if log:
        self.logger.debug(f'position: {position}, bombs: {bombs}, step {step}')

    # reset movement counter for new game
    if step == 1:
        if self.train:
            settings.move = -100
        else:
            global move
            move = -100

    # meaning of numbers
    exploding = 3
    occupied = 2
    # countdown = 1
    
    # give an index to each of four directions and current position
    top, top_pos = 0, (position[0] , position[1] - 1)
    right, right_pos = 1, (position[0] + 1, position[1])
    bottom, bottom_pos = 2, (position[0] , position[1] + 1)
    left, left_pos = 3, (position[0]  - 1, position[1])
    current = 4
    
    adjacent = np.array([occupied*abs(field[top_pos]),
                        occupied*abs(field[right_pos]),
                        occupied*abs(field[bottom_pos]),
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
    
    # information about explosion
    explosion = np.array([explosion_map[top_pos ],
                        explosion_map[right_pos],
                        explosion_map[bottom_pos ],
                        explosion_map[left_pos],
                        explosion_map[position]
                        ])
    
    info = adjacent + exploding*explosion

    # information about coin 
    collect = search_coin(self, game_state)
    if collect == top_pos and info[top] == 0:
        info[top] = -1 # go to top!
    if collect == right_pos and info[right] == 0:
        info[right] = -1
    if collect == bottom_pos and info[bottom] == 0:
        info[bottom] = -1
    if collect == left_pos and info[left] == 0:
        info[left] = -1
    
    
    # check neighboring fields for bombs
    info[top] = np.maximum(info[top], secure(top_pos, bombs, field, self))
    info[right] = np.maximum(info[right], secure(right_pos, bombs, field, self))
    info[bottom] = np.maximum(info[bottom], secure(bottom_pos, bombs, field, self))
    info[left] = np.maximum(info[left], secure(left_pos, bombs, field, self))

    test_current = secure(position, bombs, field, self)
    if test_current == 2:
        info[current] = 1
    elif test_current != 0:
        info[current] = np.maximum(info[current], test_current)
    
                
    movement = 0

    # encourage escaping into right direction 
    if info[4] in [1,3]:
        self.logger.debug(f' find escape route')
        escape = BFS_escape(field, position, bombs, self)
        self.logger.debug(f'escape : {escape}')
        if escape is not None and info[escape] == 1:
            info[escape] = 0


    # discourage bomb dropping if no escape route
    if info[current] == -1:
        self.logger.debug(f'Test escape')
        bombs.append((position, 3))
        test_escape = BFS_escape(field, position, bombs, self)
        self.logger.debug(f'escape direction : {test_escape}')
            
        if test_escape is None:
            info[current] = 0
    
    if self.train:
        movement = settings.move
    else:
        movement = move
    
    if log:
        self.logger.debug(f'adjacent : {adjacent}, explosion : {explosion}')
        self.logger.debug(f' info : {info}, move: {movement}')
    info = np.append(info, movement)

    return info.reshape(1, -1)

def secure(position, bombs, field, self):
    """
    Test if position is secure (Not in the way of a bomb)
    Return situational awareness information of a given position
    """
    
    exploding = 3
    occupied = 2
    countdown = 1

    res = occupied * abs(field[position])



    current_bomb = [0,0]
    for i in bombs:
        countdown = 1
        blocked = False

        b_pos = i[0]
        current_bomb[0] = b_pos[0] - position[0]
        current_bomb[1] = b_pos[1] - position[1]
        
        if (i[1] == 0):
            countdown = exploding


        #bombs block field
        if b_pos == position:
            res =  np.maximum(occupied, res)
            

        #check if bomb close enough
        if not np.linalg.norm(current_bomb) < settings.BOMB_TIMER:
            continue
        
        #if direct line and not stopped by wall
        if current_bomb[0] == 0:
            p = position
            #for all nodes between bomb and position, check if wall stops explosion
            for i in range(current_bomb[1]):
                if field[position[0], position[1] + i] == -1:
                    blocked = True
        if current_bomb[1] == 0:
            p = position
            #for all nodes between bomb and position, check if wall stops explosion
            for i in range(current_bomb[0]):
                if field[position[0] + 1, position[1]] == -1:
                    blocked = True
        
        if current_bomb[0] == 0 and not blocked:
            res  = np.maximum(res, countdown)
        if current_bomb[1] == 0 and not blocked:
            res  = np.maximum(res, countdown)

    #self.logger.debug(f' test secure : {position}, result: {res}')
    return res
        

    
    
def BFS_escape(field, position, bombs, self):    
    """
    Use Breadth-First-Search to find an escape route from bombs.
    Return an index of a safe step.
    """ 
    self.logger.debug(f' in BFS_escape ')
    x_occupied, y_occupied = np.where(field != 0)
    explored = [(x_occupied[i], y_occupied[i]) for i in range(len(x_occupied))]

    #bombs are 'permanent' blocks too
    for bomb in bombs:
        b_pos = bomb[0]
        explored.append(b_pos)
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[position]]

    o_top =(position[0] , position[1] - 1)
    o_right= (position[0] + 1, position[1])
    o_bottom= (position[0] , position[1] + 1)
    o_left = (position[0]  - 1, position[1])
     
    # If the desired node is
    # reached
    if secure(position, bombs, field, self) == 0:
        return 4
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
        
        #check if too far away
        if len(path) > settings.BOMB_TIMER:
            continue

        
         
        # Condition to check if the
        # current node is not visited
        
        top =(node[0] , node[1] - 1)
        right= (node[0] + 1, node[1])
        bottom= (node[0] , node[1] + 1)
        left = (node[0]  - 1, node[1])
        
        neighbours = [top, right, bottom, left]

        # account for timer of bombs
        bombs_timed =[]
        for bomb in bombs:
            if bomb[1]  - (len(path) - 1) > 0:
                bombs_timed.append((bomb[0], bomb[1]  - (len(path) - 1)))

        self.logger.debug(f'path: {path}, news bombs: {bombs_timed} ')   
        # Loop to iterate over the
        # neighbours of the node
        for neighbour in neighbours:
            if neighbour in explored:
                continue
            

            if secure(neighbour, bombs_timed, field, self) == 0:
                path.append(neighbour)
                self.logger.debug(f' shortest path: {path} ')
                if path[1] == o_top:
                    return 0
                elif path[1] == o_right:
                    return 1
                elif path[1] == o_bottom:
                    return 2
                else:
                    return 3

            
            new_path = path.copy()    
            new_path.append(neighbour)
            queue.append(new_path)
                
            # Condition to check if the
            # neighbour node is the goal
             # return shortest path and the length
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    return None


def search_coin(self, game_state):
    """
    Find the shortest path to the next coin and return the next step to the coin.

    :param dict game_state: game state information
    :return: best step to the nearest coin
    """
    field = game_state['field']
    coins = game_state['coins']
    name, score, bomb, position = game_state['self']

    # conversion from x/y coordinate of direction to index using dict
    direction2action = {
    (0, 0): ACTIONS.index('WAIT'),
    (1, 0): ACTIONS.index('RIGHT'),
    (0, 1): ACTIONS.index('DOWN'),
    (-1, 0): ACTIONS.index('LEFT'),
    (0, -1): ACTIONS.index('UP')
    }

    #print(field)
    #print(coins)
    #print(position)

    buffer = [([], position)]
    visited = np.zeros_like(field)

    while buffer:
        path, pos = buffer.pop(0)
        visited[pos[1], pos[0]] = 1

        for d_x, d_y in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            next_pos = pos + np.array([d_y, d_x])
            # print(next_pos)
            if (
                    0 < pos[0] + d_y < len(field[0]) and
                    0 < pos[1] + d_x < len(field) and
                    field[next_pos[1], next_pos[0]] != -1 and
                    not visited[next_pos[1], next_pos[0]]
            ):
                if tuple(next_pos) in coins:
                    # print(f"Coin found at {next_pos}")
                    if path:
                        return direction2action[(path[0][1], path[0][0])]
                    else:
                        return direction2action[(d_y, d_x)]

                else:
                    buffer.append((path + [(d_x, d_y)], next_pos))

    return direction2action[(0, 0)] # direction dx, dy i.e. stay

def get_closest_coin_dist(game_state): # used for reward in train.py
    """
    calculate the distance between the current position and the nearest coin

    :param dict game_state: game state information
    :return: the distance between the current position and the nearest coin
    """
    field = game_state['field']
    coins = game_state['coins']
    name, score, bomb, position = game_state['self']

    buffer = [([], position)]
    visited = np.zeros_like(field)

    while buffer:
        path, pos = buffer.pop(0)
        visited[pos[1], pos[0]] = 1

        for d_x, d_y in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            next_pos = pos + np.array([d_y, d_x])
            # print(next_pos)
            if (
                    0 < pos[0] + d_y < len(field[0]) and
                    0 < pos[1] + d_x < len(field) and
                    field[next_pos[1], next_pos[0]] != -1 and
                    not visited[next_pos[1], next_pos[0]]
            ):
                if tuple(next_pos) in coins:
                    # print(f"Coin found at {next_pos}")
                    return len(path) + 1

                else:
                    buffer.append((path + [(d_x, d_y)], next_pos))
    return np.inf