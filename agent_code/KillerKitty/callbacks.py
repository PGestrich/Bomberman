from calendar import c
import os
import pickle
import random
from cmath import inf

import time
import random

import settings

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.spatial import distance



#change in both callbacks & train!
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
dead_state = np.array([-100, -100, -100, -100, -100, -100, -100]).reshape(1, -1)
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

    if  not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = DecisionTreeRegressor(random_state=0)
        #self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)
    
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
    timestamp = time.perf_counter()

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
    if log:
        self.logger.debug(f'position: {position}, bombs: {bombs}, step {step}')


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
    
    
    #if agent stands on adjacent field, it is blocked
    #if agent stands next to adjacent field, it is dangerous
    #(if opponent drops bomb and we move into that direction, we might not have enough time to run away)
    for agent in others:
        agent_pos = agent[3]
        if agent_pos == top_pos:
            adjacent[top] = occupied
        elif distance.euclidean(agent_pos, top_pos) == 1:
            adjacent[top] = 1

        if agent_pos == right_pos:
            adjacent[right] = occupied
        elif distance.euclidean(agent_pos, right_pos) == 1:
            adjacent[right] = 1

        if agent_pos == bottom_pos:
            adjacent[bottom] = occupied
        elif distance.euclidean(agent_pos, bottom_pos) == 1:
            adjacent[bottom] = 1


        if agent_pos == left_pos:
            adjacent[left] = occupied
        elif distance.euclidean(agent_pos, left_pos) == 1:
            adjacent[left] = 1
                        
    #explosions are dangerous too
    explosion = np.array([explosion_map[top_pos ],
                        explosion_map[right_pos],
                        explosion_map[bottom_pos ],
                        explosion_map[left_pos],
                        explosion_map[position]
                        ])
    info = adjacent + exploding*explosion
    info = info.astype(int)
    
    
    
    #check neighboring fields for bombs
    info[top] = np.maximum(info[top], secure(top_pos, bombs, field, self))
    info[right] = np.maximum(info[right], secure(right_pos, bombs, field, self))
    info[bottom] = np.maximum(info[bottom], secure(bottom_pos, bombs, field, self))
    info[left] = np.maximum(info[left], secure(left_pos, bombs, field, self))

    test_current = secure(position, bombs, field, self)
    if test_current == 2:
        info[current] = 1
    elif test_current != 0:
        info[current] = np.maximum(info[current], test_current)
    
                
    #movement = 0

    #encourage escaping into right direction 
    if info[4] in [1,3]:
        #self.logger.debug(f' find escape route')
        escape = BFS_escape(field, position, bombs, others, self, False)
        self.logger.debug(f'escape : {escape}')
        if escape is not None and info[escape] in [1,3]:
            info[escape] = 0


    # discourage bomb dropping if no escape rout:
    if info[current] == -1:
        #self.logger.debug(f'Test escape')
        bombs.append((position, 3))
        test_escape = BFS_escape(field, position, bombs, others, self, True)
        self.logger.debug(f'possible escape : {test_escape}')
            
        if test_escape is None:
            info[current] = 0
    
    #include info about opponents/coins and crates
    coin, opponent = BFS_opponent(field, position, others, coins,  self)
    crate = BFS_crate(field, position, self)
    
    
    if log:
        self.logger.debug(f'adjacent : {adjacent}, explosion : {explosion}')
        self.logger.debug(f' info : {info}, coin: {coin}, crate:{crate}, opponent: {opponent}')
        self.logger.debug(f' time: {time.perf_counter() - timestamp}')
    info = np.append(info, [coin, crate, opponent])
    
    return info.reshape(1, -1)

def secure(position, bombs, field, self):
    """
    Test if position is secure (Not in the way of a bomb)
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
        
        if (i[1] <= 0):
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
        

    
    
def BFS_escape(field, position, bombs, oponents, self, test):    
    """
    Use Breadth-First-Search to find an escape rout from bombs
    """
    self.logger.debug(f' in BFS_escape ')
    #self.logger.debug(f'field \n  {field}')

    x_occupied, y_occupied = np.where(field != 0)
    explored = [(x_occupied[i], y_occupied[i]) for i in range(len(x_occupied))]

    #agents block too
    for agent in oponents: 
        explored.append(agent[3])

    #simple trap preventing:
    #all agents that can drop a bomb will do so next step
    if test:
        for agent in oponents:
            if agent[2]:
                bombs.append((agent[3],3))

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
     
    # If the desired node is reached
    if secure(position, bombs, field, self) == 0:
        return 4
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
        self.logger.debug(f'current {path}')
        
        if node != position and node in explored:
            continue
        
            
        
        
        #check if too far away
        if len(path) > settings.BOMB_TIMER:
            continue

        
        
        
        top =(node[0] , node[1] - 1)
        right= (node[0] + 1, node[1])
        bottom= (node[0] , node[1] + 1)
        left = (node[0]  - 1, node[1])
        
        neighbours =[top, right, bottom, left]
        np.random.shuffle(neighbours)
        #neighbours = neighbours[np.random.permutation(4)]
        #print(neighbours)
        
        # account for timer of bombs (only simple approximation of explosions by using explosion timer)
        bombs_timed =[]
        for bomb in bombs:
            if bomb[1]  - (len(path) - 1) > - settings.EXPLOSION_TIMER:
                bombs_timed.append((bomb[0], bomb[1]  - (len(path) - 1)))

        
        new_paths = []
        discard = False
        # Loop to iterate over the
        # neighbours of the node
        for neighbour in neighbours:
            
            #discard if blocked
            if neighbour in explored:
                continue
            
            #if safe, return the first step
            safe = secure(neighbour, bombs_timed, field, self)
            if  safe == 0:
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
            #don't escape over exploding tiles
            elif safe == 3:
                continue
            

            

            
            new_path = path.copy()    
            new_path.append(neighbour)
            new_paths.append(new_path)
            #new_paths.append(new_path)
            
        
        explored.append(node)    
        if not discard and new_paths != []:
            for p in new_paths:
                queue.append(p)
                
    return None



def BFS_opponent(field, position, others, coins, self):    
    """
    Use Breadth-First-Search to find opponents and coins
    """
    #self.logger.debug(f' in BFS_opponent ')
    self.logger.debug(f' coins: {coins} ')
    timestamp = time.perf_counter()
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

    coin = -100
    opponent = -100
    found_opponent = False
    found_coin = False
     
    #if no coins available, don't search for them
    if coins == []:
        found_coin = True

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


        # Loop to iterate over the
        # neighbours of the node
        for neighbour in neighbours:
            if neighbour in explored:
                continue
            
            #check if found coin - remember first step
            if not found_coin and neighbour in coins:
                path.append(neighbour)
                if path[1] == o_top:
                    coin = 0
                elif path[1] == o_right:
                    coin = 1
                elif path[1] == o_bottom:
                    coin = 2
                else:
                    coin = 3
                found_coin = True
            

            #check for opponents - remember first step
            for agent in others:
                if found_opponent or agent[3] != neighbour:
                    continue
                
                path.append(neighbour)
                if path[1] == o_top:
                    opponent = 0
                elif path[1] == o_right:
                    opponent = 1
                elif path[1] == o_bottom:
                    opponent = 2
                else:
                    opponent = 3
                
                found_opponent = True

            #stop here if found both
            if found_coin and found_opponent:
                return coin, opponent
            #else continue search
            new_path = path.copy()    
            new_path.append(neighbour)
            queue.append(new_path)

            
        explored.append(node)
 
    
    #if opponent found, but no coins: return oponent
    #if nothing found, return original value (-100)
    return coin, opponent

def BFS_crate(field, position, self): 

    """
    Use Breadth-First-Search to find crates
    """   
    #self.logger.debug(f' in BFS_crate ')
    timestamp = time.perf_counter()
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


        # Loop to iterate over the
        # neighbours of the node
        for neighbour in neighbours:
            
            #check if we found crate
            if field[neighbour] != 1:
                new_path = path.copy()    
                new_path.append(neighbour)
                queue.append(new_path)
                continue
                
            path.append(neighbour)
            res = 0
            if path[1] == o_top:
                res =  0
            elif path[1] == o_right:
                res =  1
            elif path[1] == o_bottom:
                res = 2
            else:
                res = 3


            return res        
        explored.append(node)
 
   #no crate found
    return -100