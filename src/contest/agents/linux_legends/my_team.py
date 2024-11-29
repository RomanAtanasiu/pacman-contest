# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import math
import random
import util
import os
#import numpy as np

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

#aproximate q-learing using linear function approximation
class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def __init__(self, index, alpha=0.01, gamma=0.85, epsilon=0.0, ghost_threshold=5, max_distance=100, max_food=20): 
        super().__init__(index)
        print("Initializing OffensiveReflexAgent\n")
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.theta = util.Counter()  # Weights

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Calculate the distance from the bottom-right corner to the top-left corner
        walls = game_state.get_walls()
        bottom_right = (walls.width - 1, 0)
        top_left = (0, walls.height - 1)
        self.max_distance = walls.width * (walls.height +1) 
        # Calculate the initial number of food pellets
        self.max_food = len(self.get_food(game_state).as_list())
        
        
    #added features: remaining food, num_carrying_food, distance_to_boundary, distance_to_rival, ghost_in_threshold, ghost_in_trajectory
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)/self.max_food  # self.get_score(successor)
        my_pos = successor.get_agent_state(self.index).get_position()

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = -min_distance/self.max_distance
        else:
            features['distance_to_food'] = 0
            
            
        # if you are carring too much food is bad, if you are carrying little food is good
        carried_food = game_state.get_agent_state(self.index).num_carrying
        if carried_food <= 3:
            features['num_carrying_food'] = (carried_food)/self.max_food
        else:
            features['num_carrying_food'] = (-carried_food)/self.max_food
        
        
        if carried_food > 3:
            #compute distance to enemy territory
            walls = game_state.get_walls()
            mid_x = walls.width // 2
            # Determine the boundary x-coordinate based on the team
            if self.red:
                boundary_x = mid_x - 1
            else:
                boundary_x = mid_x
            # Find all boundary points
            boundary_points = [(boundary_x, y) for y in range(walls.height) if not walls[boundary_x][y]]
            # Compute the distance to the closest boundary point
            min_distance = min([self.get_maze_distance(my_pos, point) for point in boundary_points])
            features['distance_to_boundary'] = -((carried_food/self.max_food)*min_distance)/self.max_distance #more food carrying is worse
        else:
            features['distance_to_boundary'] = 0
            
        # Compute distance to the closest rival agent as pacman
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        rivals = [a for a in enemies if a.get_position() is not None]
        if len(rivals) > 0:
            dists = [self.get_maze_distance(my_pos, rival.get_position()) for rival in rivals]
            if game_state.get_agent_state(self.index).is_pacman:           
                features['distance_to_rival'] = (min(dists)*0.7)/self.max_distance #root because if far is not that bad
            else:
                features['distance_to_rival'] = 0
        else:
            features['distance_to_rival'] = 0

        """
        # Compute if a ghost is closer than a threshold distance to Pacman
        ghost_threshold = 4
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if len(ghosts) > 0:
            ghost_dists = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            if min(ghost_dists) <= ghost_threshold:
                features['ghost_in_threshold'] = -1
            else:
                features['ghost_in_threshold'] = 0
        else:
            features['ghost_in_threshold'] = 0
    """
        # Is a ghost in trajctory to food?
        # Compute the angle to the closest food
        """
        if len(food_list) > 0:
            closest_food = None
            min_distance = float('inf')
            for food in food_list:
                distance = self.get_maze_distance(my_pos, food)
                if distance < min_distance:
                    min_distance = distance
                    closest_food = food
            delta_x = closest_food[0] - my_pos[0]
            delta_y = closest_food[1] - my_pos[1]
            angle_to_food = math.atan2(delta_y, delta_x)
        # Compute the angle to the closest ghost
        elif len(ghosts) > 0:
            closest_ghost = None
            min_distance = float('inf')
            for ghost in ghosts:
                distance = self.get_maze_distance(my_pos, ghost.get_position())
                if distance < min_distance:
                    min_distance = distance
                    closest_ghost = ghost
            delta_x = closest_ghost.get_position()[0] - my_pos[0]
            delta_y = closest_ghost.get_position()[1] - my_pos[1]
            angle_to_ghost = math.atan2(delta_y, delta_x)
            # Check if the angle to the ghost is similar to the angle to the food
            if abs(angle_to_food - angle_to_ghost) < math.pi / 4:
                features['ghost_in_trajectory'] = 1
            else:
                features['ghost_in_trajectory'] = 0
        else:
            features['ghost_in_trajectory'] = 0
        """
        return features
    
    
    
    
    def reward_function(self, game_state, action):
        reward = 0
        self.load_weights() #load the weights from the file
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
    
        # eat food
        previous_state = game_state.get_agent_state(self.index)
        if successor.get_agent_state(self.index).num_carrying > previous_state.num_carrying:
            reward += 3
    
        # eat capsule
        capsule_list = self.get_capsules(game_state)
        if my_pos in capsule_list:
            reward += 20
    
    
    
        #  being captured
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if my_state.is_pacman and my_pos in [ghost.get_position() for ghost in ghosts]:
            if my_state.num_carrying > 0: #if you are carrying food, you should not be captured
                reward -= 15*successor.num_carrying
            else:
                reward -= 10
    
        
        # Reward for approaching the boundary if you are a ghost
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        if self.red:
            boundary_x = mid_x - 1
        else:
            boundary_x = mid_x
        boundary_points = [(boundary_x, y) for y in range(walls.height) if not walls[boundary_x][y]]
        min_distance = min([self.get_maze_distance(my_pos, point) for point in boundary_points])
        previous_pos = previous_state.get_position()
        previous_min_distance = min([self.get_maze_distance(previous_pos, point) for point in boundary_points])
        if not my_state.is_pacman:
            if min_distance < previous_min_distance:
                reward += 1
                
        """
        # Reward for approaching the boundary if carrying food
        if my_state.is_pacman and my_state.num_carrying > 0:
            if min_distance < previous_min_distance:
                reward += 3*my_state.num_carrying
        """
        # penalti if you stay as ghost (you are an offensive agent, you should be pacman as long as possible)
        previous_state = game_state.get_agent_state(self.index)
        if not previous_state.is_pacman and not my_state.is_pacman:
            reward -= 30
    
        # getting points
        if my_state.num_returned > previous_state.num_returned:
            reward += 20 * (my_state.num_returned - previous_state.num_returned)
    
        return reward/200
    
    def evaluate(self, game_state, action):
        weights = self.get_weights(game_state, action)
        features = self.get_features(game_state, action)
        return sum(weights[f] * features[f] for f in features)

    
    def choose_action(self, game_state):
        
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()


        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            chosen_action = random.choice(actions)  # choose a random action
        else:
            q_values = [self.evaluate(game_state, action) for action in actions]
            max_q_value = max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q_value]
            chosen_action = random.choice(best_actions)  # choose one of the best actions randomly

        # Obtain the sucesor state
        successor = self.get_successor(game_state, chosen_action)
        # Calculte the current features
        current_features = self.get_features(game_state, chosen_action)
        # Calculate the reward
        reward = self.reward_function(game_state, chosen_action)
        # Calculate the current Q-value
        current_q_value = self.evaluate(game_state, chosen_action)

        # find the best q-value for the next state
        successor_actions = successor.get_legal_actions(self.index)
        successor_q_values = [self.evaluate(successor, a) for a in successor_actions]
        max_successor_q_value = max(successor_q_values) if successor_q_values else 0

        # Calculate the TD-error
        delta = reward + self.gamma * max_successor_q_value - current_q_value

        # Update the weights
        for f in current_features:
            self.theta[f] += self.alpha * delta * current_features[f]

        self.save_weights()
        return chosen_action
    
    
    def get_weights(self, game_state, action):
        return self.theta
    
    
    def load_weights(self, file_path='./agents/linux_legends/weights.json'):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                weights_str = file.read()
                weights_str = weights_str.replace('inf', 'float("inf")').replace('nan', 'float("nan")')
                self.theta = util.Counter(eval(weights_str))
        else:
            print("No weight file found. Initializing weights to zeros.")
            self.theta = util.Counter()
    
    def save_weights(self, file_path='weights.json'):
        with open(file_path, 'w') as f:
            f.write(str(dict(self.theta)))


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

class AStarAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.previous_food = None
        self.previous_capsules = None
        self.target_position = None
        self.current_position = None
        self.count_actions = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.current_position = self.start
        CaptureAgent.register_initial_state(self, game_state)
        self.previous_food = self.get_food_you_are_defending(game_state).as_list()

        #S'ha de pensar q fer amb les capsules
        self.previous_capsules = self.get_capsules_you_are_defending(game_state)

    def choose_action(self, game_state):
        # Get the food you are defending and the capsules you are defending in this state
        current_food = self.get_food_you_are_defending(game_state).as_list()
        current_capsules = self.get_capsules_you_are_defending(game_state)

        # Check if the food or capsules have changed
        if current_food != self.previous_food or current_capsules != self.previous_capsules:
            # get the food position that have been eaten
            eaten_food = [food for food in self.previous_food if food not in current_food]

            # get the capsules that have been eaten
            eaten_capsules = [capsule for capsule in self.previous_capsules if capsule not in current_capsules]

            # in each new state, the maximum food eaten by the enemies is two because there are two enemies
            # if the food has been eaten, update the target position to the nearest eaten food
            if eaten_food:
                self.target_position = min(eaten_food, key=lambda x: self.get_maze_distance(self.start, x))

        # Check if an enemie is visible
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]



        # if there are invaders, update the target position towards the nearest invader
        if invaders:
            self.target_position = min([invader.get_position() for invader in invaders], key=lambda x: self.get_maze_distance(self.start, x))

        # update the previous food and capsules
        self.previous_food = current_food
        self.previous_capsules = current_capsules

        # if there is a target position, call a_star_search to get the path to the target position

        if self.target_position:
            target = self.target_position
        else:
            center = (game_state.data.layout.width // 2, game_state.data.layout.height // 2)
            if current_food:
                target = min(current_food, key=lambda x: self.get_maze_distance(center, x))
            else:
                target = center

        current_position = game_state.get_agent_position(self.index)
        path = self.a_star_search(game_state, current_position, target)

        # reset the target position
        self.count_actions += 1
        # guardar l'ultima posicio del fantasma vist o menjar menjat durant 10 posicions pq vagi cap allà
        if self.count_actions == 25:
            self.target_position = None
            self.count_actions = 0


        # if there is a path, return the first action to the next position in the path
        if path:
            print(path)
            return path[0]
        else:
            # if there is no path, return a random action
            # si esta al centre del taulell, no pot anar cap a west
            actions = game_state.get_legal_actions(self.index)
            if current_position[0] == game_state.data.layout.width // 2:
                actions = [action for action in actions if action != Directions.WEST]
            return random.choice(actions)

    def a_star_search(self, game_state, start, goal):
        """
        A* search
        """
        frontier_priority_queue = util.PriorityQueue()
        frontier_priority_queue.push((start, []), 0)
        expanded_nodes = set()

        while not frontier_priority_queue.is_empty():
            current_pos, path = frontier_priority_queue.pop()

            if current_pos in expanded_nodes:
                continue

            expanded_nodes.add(current_pos)

            if current_pos == goal:
                return path

            for next_pos, direction, cost in self.get_successors(game_state, current_pos):
                if next_pos not in expanded_nodes:
                    new_path = path + [direction]
                    new_cost = len(new_path) + self.get_maze_distance(next_pos, goal)
                    frontier_priority_queue.push((next_pos, new_path), new_cost)

        return []

    def get_successors(self, game_state, position):
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            next_position = self.get_successor_position(position, direction)
            if not game_state.has_wall(int(next_position[0]), int(next_position[1])):
                successors.append((next_position, direction, 1))
        return successors

    def get_successor_position(self, position, direction):
        x, y = position
        if direction == Directions.NORTH:
            return (x, y + 1)
        elif direction == Directions.SOUTH:
            return (x, y - 1)
        elif direction == Directions.EAST:
            return (x + 1, y)
        elif direction == Directions.WEST:
            return (x - 1, y)
        return position