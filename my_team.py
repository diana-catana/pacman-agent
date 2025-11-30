
from contest.capture_agents import CaptureAgent
import contest.util as util
import random, time
from contest.util import nearest_point

from contest.game import Directions, Actions
import contest.game
import json

from collections import deque

def nearest_free(game_state, start):
    """Return nearest non-wall tile using BFS."""
    w = game_state.get_walls().width
    h = game_state.get_walls().height
    q = deque([start])
    visited = {start}
    while q:
        x, y = q.popleft()
        if 0 <= x < w and 0 <= y < h and not game_state.has_wall(x, y):
            return (x, y)
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return start

def choose_best_offset(game_state, entry_points, is_red):
    """Automatically choose inward distance (1,2,3) depending on choke depth."""
    best_offset = 1
    best_valid = -1
    
    for offset in [1, 2, 3]:
        dx = -offset if is_red else offset
        inward = [(x+dx, y) for x, y in entry_points]
        valid = sum(1 for (x,y) in inward if not game_state.has_wall(x,y))
        if valid > best_valid:
            best_valid = valid
            best_offset = offset

    return best_offset

def get_defensive_checkpoints(game_state, index):
    """
    Returns (lower_checkpoint, upper_checkpoint) for ANY layout.
    Automatically:
      - finds entry tunnels
      - detects inward choke depth
      - places checkpoints correctly
      - avoids walls
    """

    walls = game_state.get_walls()
    w, h = walls.width, walls.height
    is_red = game_state.is_on_red_team(index)

    # ----- 1. find midline entry points -----
    mid_x = w//2 - 1 if is_red else w//2
    entry_points = [(mid_x, y) for y in range(h)
                    if not game_state.has_wall(mid_x, y)]

    if not entry_points:
        # fallback: shouldn't happen on legal layouts
        mid_y = h//2
        return ( (mid_x, mid_y-1), (mid_x, mid_y+1) )

    # ----- 2. dynamically choose best inward offset -----
    offset = choose_best_offset(game_state, entry_points, is_red)
    dx = -offset if is_red else offset

    inward_points = [(x+dx, y) for x, y in entry_points]

    # ----- 3. split into upper / lower groups -----
    inward_points.sort(key=lambda p: p[1])
    mid = len(inward_points)//2

    lower_group = inward_points[:mid]
    upper_group = inward_points[mid:]

    # Safety fallback if one group ends empty
    if len(lower_group) == 0: lower_group = upper_group
    if len(upper_group) == 0: upper_group = lower_group

    # ----- 4. centroid of each group -----
    def centroid(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs)//len(xs), sum(ys)//len(ys))

    lower_cp = centroid(lower_group)
    upper_cp = centroid(upper_group)

    # ----- 5. BFS shift off a wall -----
    lower_cp = nearest_free(game_state, lower_cp)
    upper_cp = nearest_free(game_state, upper_cp)

    return lower_cp, upper_cp

def nearest_free(game_state, start):
    """Return nearest non-wall tile using BFS."""
    w = game_state.get_walls().width
    h = game_state.get_walls().height
    q = deque([start])
    visited = {start}
    while q:
        x, y = q.popleft()
        if 0 <= x < w and 0 <= y < h and not game_state.has_wall(x, y):
            return (x, y)
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return start


def choose_best_offset(game_state, entry_points, is_red):
    best_offset = 1
    best_valid = -1

    for offset in [1, 2, 3]:
        dx = -offset if is_red else offset
        inward = [(x+dx, y) for x, y in entry_points]
        valid = sum(1 for (x,y) in inward if not game_state.has_wall(x, y))
        if valid > best_valid:
            best_valid = valid
            best_offset = offset

    return best_offset

def get_split_defensive_checkpoints(game_state, index):
    """
    Returns FOUR defensive checkpoints:
        - upper_defender:   (upper_cp, mid_upper_cp)
        - lower_defender:   (mid_lower_cp, lower_cp)

    Automatically:
      - finds entry tunnels
      - detects inward choke depth
      - divides region evenly
      - relocates checkpoints if walls exist
    """

    walls = game_state.get_walls()
    w, h = walls.width, walls.height
    is_red = game_state.is_on_red_team(index)

    # ---------- 1. Find crossing entry tiles ----------
    mid_x = w//2 - 1 if is_red else w//2
    entry_points = [(mid_x, y) for y in range(h)
                    if not game_state.has_wall(mid_x, y)]

    if not entry_points:
        # fallback (almost never happens)
        return (None, None, None, None)

    # ---------- 2. Choose inward choke depth ----------
    offset = choose_best_offset(game_state, entry_points, is_red)
    dx = -offset if is_red else offset
    inward_points = [(x+dx, y) for x, y in entry_points]

    # ---------- 3. Split into quarters ----------
    inward_points.sort(key=lambda p: p[1])

    n = len(inward_points)
    q1 = inward_points[: n//4] or [inward_points[0]]
    q2 = inward_points[n//4 : n//2] or [inward_points[n//4]]
    q3 = inward_points[n//2 : 3*n//4] or [inward_points[n//2]]
    q4 = inward_points[3*n//4 :] or [inward_points[-1]]

    # Centroid helper
    def centroid(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs)//len(xs), sum(ys)//len(ys))

    # ---------- 4. Compute 4 anchors ----------
    upper_cp      = centroid(q1)
    mid_upper_cp  = centroid(q2)
    mid_lower_cp  = centroid(q3)
    lower_cp      = centroid(q4)

    # ---------- 5. BFS shift away from walls ----------
    upper_cp      = nearest_free(game_state, upper_cp)
    mid_upper_cp  = nearest_free(game_state, mid_upper_cp)
    mid_lower_cp  = nearest_free(game_state, mid_lower_cp)
    lower_cp      = nearest_free(game_state, lower_cp)

    return [upper_cp, mid_upper_cp, mid_lower_cp, lower_cp]


def create_team(first_index, second_index, is_red,  first = 'Agent2', second = 'Agent1', num_training = 10, **args):
  return [eval(first)(first_index), eval(second)(second_index)]


class DDagent(CaptureAgent):



  def rec(self,i,j,depth_map,visited,h):
    sum = 0
    visited[h-i-1][j] = True
    direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy,dx in direction:
      y = i + dy
      x = j + dx
      if visited[h-y-1][x] == False and depth_map[h-y-1][x] == 3:
        self.dead_ends_depth[h-y-1][x] = self.rec(y,x,depth_map,visited,h)
        sum = self.dead_ends_depth[h-y-1][x]
    return sum + 1

  def register_initial_state(self, game_state):
    self.enemy_cost = 1.5
    self.own_cost = 1.0
    self.layout_width = game_state.data.layout.width

    w = game_state.data.layout.width
    h = game_state.data.layout.height


    # Initialize state maps
    self.dead_ends_depth = [[0] * w for _ in range(h)]
    depth_map = [[0] * w for _ in range(h)]
    visited = [[False] * w for _ in range(h)]
    walls = game_state.get_walls()
    self.walls = walls
    queue = util.Queue()
    # Start exploration from each unvisited, non-wall cell
    for i in range(h):
        for j in range(w):
            if not game_state.has_wall(j, i):
                neighbors = Actions.get_legal_neighbors((j, i), walls)
                depth_map[h-i-1][j] = len(neighbors)
                if len(neighbors) == 2:
                  queue.push((i, j))
    
    while not queue.is_empty():
      i,j = queue.pop()
      self.dead_ends_depth[h-i-1][j] = self.rec(i,j,depth_map,visited,h)

    self.max_dead_end = max(value for row in self.dead_ends_depth for value in row)

    # self.initial_weight()
    self.total_food = len(self.get_food(game_state).as_list())
    width = game_state.data.layout.width
    height = game_state.data.layout.height
    self.max_distance = (width**2 + height**2)**0.5

    neighbors = []
    if game_state.is_on_red_team(self.index):
      neighbors = [(0, 1), (0, -1), (-1, 1), (-1, 0), (-1,-1) ]
    else:
      neighbors = [(0, 1), (0, -1), (1, 1), (1, 0), (1,-1) ]

    self.weights_array = self.get_weights()
    self.initial_weight()


    # LOWER CHECK POINT
    temp_h = max(1, h - 3)
    if game_state.is_on_red_team(self.index):
      self.defence_ancor_point1 = (w // 2 - 2, temp_h)
    else:
      self.defence_ancor_point1 = (w // 2 + 1, temp_h)

    if game_state.has_wall(self.defence_ancor_point1[0], self.defence_ancor_point1[1]):
      dist = 1
      while game_state.has_wall(self.defence_ancor_point1[0], self.defence_ancor_point1[1]):
        for dx,dy in neighbors:
          self.defence_ancor_point1 = (self.defence_ancor_point1[0] + dist*dx, self.defence_ancor_point1[1] + dist * dy)
          if not game_state.has_wall(self.defence_ancor_point1[0], self.defence_ancor_point1[1]):
            break
        dist += 1
    self.default_defence_ancor_point1 = self.defence_ancor_point1

    # UPPER CHECK POINT
    temp_h = min(h-2 , 3)
    if game_state.is_on_red_team(self.index):
      self.defence_ancor_point2 = (w // 2 - 2, temp_h)
    else:
      self.defence_ancor_point2 = (w // 2 + 1, temp_h)

    if game_state.has_wall(self.defence_ancor_point2[0], self.defence_ancor_point2[1]):
      dist = 1
      while game_state.has_wall(self.defence_ancor_point2[0], self.defence_ancor_point2[1]):
        for dx,dy in neighbors:
          self.defence_ancor_point2 = (self.defence_ancor_point2[0] + dist*dx, self.defence_ancor_point2[1] + dist * dy)
          if not game_state.has_wall(self.defence_ancor_point2[0], self.defence_ancor_point2[1]):
            break
        dist += 1
    self.default_defence_ancor_point2 = self.defence_ancor_point2
    
    self.defence_ancor_point = self.default_defence_ancor_point2
    
    # self.default_defence_ancor_point1, self.default_defence_ancor_point2 = get_defensive_checkpoints(game_state, self.index)
    self.double_anchors = get_split_defensive_checkpoints(game_state, self.index)
    self.mode = "ATTACK"


        

    self.start = game_state.get_agent_position(self.index)
    CaptureAgent.register_initial_state(self, game_state)

  def evaluate(self, game_state, action):
    features = self.get_features(game_state, action)
    # print(self.index, action, features)
    # print()
    return features * self.weights

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

  def choose_action(self, game_state):
    # print(self.index, self.current_mode)
    legal_actions = game_state.get_legal_actions(self.index)
    if len(legal_actions) == 0:
      return None
    action_vals = {}
    best_value = float('-inf')
    for action in legal_actions:
      value = self.evaluate(game_state, action)
      action_vals[action] = value
      if value > best_value:
        best_value = value
    best_actions = [k for k, v in action_vals.items() if v == best_value]
    # return random.choice(legal_actions)
    return random.choice(best_actions)
  
  def get_features(self, game_state, action):
    features = util.Counter()
    successor = self.get_successor(game_state, action)
    food_to_eat = self.get_food(game_state).as_list()
    my_pos = successor.get_agent_state(self.index).get_position()
    teammate_index = (self.index + 2) % 4

    walls = game_state.get_walls()
    enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
    ghosts= [a for a in enemies if not a.is_pacman and a.get_position() != None]
    enemy_pacman = [a.get_position() for a in enemies if a.is_pacman and a.get_position() != None]
    delete = [a.get_position() for a in enemies if a.is_pacman]
    # print(len(delete)- len(enemy_pacman))
    agent_position = game_state.get_agent_position(self.index)
    x, y = agent_position
    
    next_x, next_y = int(my_pos[0]), int(my_pos[1])

    # DEFENCIVE
   
    # ATACK
    # FOOD
    if len(food_to_eat) > 0:  # This should always be True,  but better safe than sorry
        min_distance = min([self.get_maze_distance(my_pos, food) for food in food_to_eat])
        diagonal = (walls.width**2 + walls.height**2)**0.5
        # diagonal is the furthest that the food can be and in a way we normalize the [0,1] 
        features["closest-food"] = float(min_distance) / diagonal
    
    capsules = self.get_capsules(successor)    
    if len(capsules) > 0:
      closest_cap = min([self.get_maze_distance((next_x, next_y), cap) for cap in capsules])
      # MAXIMIZE THIS
      features["closest_capsule"] = (self.max_distance - closest_cap) / self.max_distance
      
    # closest_enemy_ghost
    if len(ghosts) > 0:
      min_ghost = min(ghosts, key=lambda g: self.get_maze_distance((next_x, next_y), g.get_position()))
      closest_enemy = self.get_maze_distance((next_x, next_y), min_ghost.get_position())
      
      if len(capsules) > 0:
        min_ghost_to_capsule = min([self.get_maze_distance(min_ghost.get_position(), cap) for cap in capsules])
        
      # scared enemies
      if min_ghost.scared_timer > 0 and min_ghost.scared_timer < 21: # less then 5 more moves of them beeing scared
        # MINIMIZE THIS (you want to eat the ghosts)
        features["closest_enemy_ghost"] = -1 * (6 - closest_enemy) / 6
      elif  min_ghost.scared_timer >= 21:
        features["closest_enemy_ghost"] = 0
      elif closest_enemy <= 5 and len(capsules) > 0 and min_ghost_to_capsule > closest_cap:
          features["closest_capsule"] *=10000
          features["closest-food"] = 0
          features["closest_enemy_ghost"] = 0
      elif closest_enemy <= 5:
        # MINIMIZE THIS
        features["closest_enemy_ghost"] = (6 - closest_enemy) / 6
        features["closest-food"] = 0
      elif closest_enemy == 0:
        features["dead"] = 1
        
        
        

    # BOTH ATTACK AND DEFEND WILL CHASE AFTER THE OPPONENT PACMAN
    # JUST WITH DIFFERENT WEIGHTS (FOR ATTACKER IS LESS IMPORTANT TO CATCH THE ENEMY)
    # closest_enemy_pacman
    if len(enemy_pacman) > 0:
      closest_enemy = min([self.get_maze_distance((next_x, next_y), e) for e in enemy_pacman])
      if closest_enemy <= 5 and game_state.get_agent_state(self.index).scared_timer == 0:
        # poveke
        features["closest_enemy_pacman"] = (6 - closest_enemy) / 6
    
    # vertical distance to enemy VERTICAL    
    if len(enemy_pacman) > 0 :
      closest_enemy = min([abs(next_y - e[1]) for e in enemy_pacman])
      if closest_enemy <= 5 and game_state.get_agent_state(self.index).scared_timer == 0:
        # poveke
        features["closest_enemy_pacman_vertical"] = (6 - closest_enemy) / 6



    # MINIMIZE THIS
    features["food_left"] = len(food_to_eat) / self.total_food

    # MAXIMIZE THIS
    score = self.get_score(successor)
    features["score"] = (self.get_score(successor) + self.total_food) / (self.total_food * 2)
    
    # BOTH DEFEND IF WINNING
    if self.get_score(game_state) > 0:
      self.weights = self.weights_array[0]  # defende weights
      self.current_mode = "DOUBLE_DEFENCE"
    else:
      # this will set the weights to attack and defend instead of both defend
      self.initial_weight()

    
    closest_way_home = 0
    if game_state.get_agent_state(self.index).is_pacman:
      # pomalo
      features["num_carring"] = game_state.get_agent_state(self.index).num_carrying / self.total_food
      middle = walls.width // 2 - 1 if  self.red else walls.width // 2
      closest_way_home = min([self.get_maze_distance((next_x, next_y), (middle, y)) for y in range(walls.height) if not game_state.has_wall(middle, y)])
      features["num_carring"] = features["num_carring"]  * closest_way_home / self.max_distance
      if features["num_carring"] + score > 0:
        # if with the food you are winning then "go home"
        features["num_carring"] *=100
  
      if self.dead_ends_depth[next_y][next_x] > 0:
        features["dead_end"] = self.dead_ends_depth[next_y][next_x] / self.max_dead_end
      if len(enemy_pacman) > 0 and closest_enemy < 5:
        features["dead_end"] *= 100
      # print("DEAD END:", features["dead_end"],self.dead_ends_depth[next_y][next_x])

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1


    if game_state.get_agent_state(self.index).is_pacman:
      features["defence"] = 0
    else:
      features["defence"] = 1
    
    old_game_state = self.get_previous_observation()
    if old_game_state is  not None:
      food_old = self.get_food_you_are_defending(old_game_state).as_list()
      food_new = self.get_food_you_are_defending(game_state).as_list()

      missing_food = list(set(food_old) - set(food_new))
      # print(missing_food)

      if len(missing_food) > 0:
        self.defence_ancor_point = missing_food[0]


    # print(walls.width, walls.height)
    if self.current_mode == "DEFENCE":
      anchor1 = self.default_defence_ancor_point1
      anchor2 = self.default_defence_ancor_point2
        
    if self.current_mode == "DOUBLE_DEFENCE":
      # if smaller then get anchor 0 and 1 else get 2 and 3
      anchor1 = self.double_anchors[0] if self.index < teammate_index else self.double_anchors[2]
      anchor2 = self.double_anchors[1] if self.index < teammate_index else self.double_anchors[3]      
      
    if self.current_mode == "DEFENCE" or self.current_mode == "DOUBLE_DEFENCE":
      # DEFENSIVE AGENT
      pacman_unknown_position= [a for a in enemies if a.is_pacman]
      if len(pacman_unknown_position) == 0 and self.defence_ancor_point != anchor1 and self.defence_ancor_point != anchor2:
        dist_ap1 = self.get_maze_distance((x, y), anchor1)
        dist_ap2 = self.get_maze_distance((x, y), anchor2)
        if dist_ap1 < dist_ap2:
          self.defence_ancor_point = anchor1
        else:
          self.defence_ancor_point = anchor2      
          
      if (x,y) == anchor1:
        self.defence_ancor_point = anchor2
      elif (x,y) == anchor2:
        self.defence_ancor_point = anchor1
      # IF WE ARE SCARED DO NOT RUN AWAY FROM ENEMY PACMAN
      if game_state.get_agent_state(self.index).scared_timer > 0:
        features["closest_enemy_pacman"] = 0
        features["closest_enemy_pacman_vertical"] = 0
        # features["defence"] = 0
        features["defence_ancor_point"] = 0
        # features["reverse"] = 0
      else:
        features["closest_enemy_pacman"] *= 40
        features["closest_enemy_pacman_vertical"] *= 40
        features["closest_capsule"] = 0
        features["closest-food"] = 0
        # features["closest_enemy_ghost"] = 0
        features["num_carring"] = 0
        features["dead_end"] = 0
        features["stop"] *= 3
  
      
    elif self.current_mode == "ATTACK":
      # OFFENSIVE AGENT
      if len(enemy_pacman) == 0 or  min([self.get_maze_distance((next_x, next_y), e) for e in enemy_pacman]) == 0:
        self.mode = "ATTACK"


      team_mate_position = game_state.get_agent_state(teammate_index).get_position()
      between_team_mate_distance = self.get_maze_distance((next_x, next_y), team_mate_position)

      if  len(enemy_pacman) > 0 and between_team_mate_distance > 5:
        closest_enemy = min([self.get_maze_distance((next_x, next_y), e) for e in enemy_pacman])
        closest_enemy_pacman_to_team_mate = min([self.get_maze_distance(team_mate_position, e) for e in enemy_pacman])
        if closest_enemy  <= 5 and closest_enemy_pacman_to_team_mate < 5:
          self.mode = "DEFFENSIVE"
      if closest_way_home + 3 >= game_state.data.timeleft / 4:
        self.mode = "TIMEOUT"
    
      # mode where you try to suround the enemy from both sides
      if self.mode == "DEFFENSIVE":
        features["closest-food"] = 0
        features["closest_enemy_ghost"] = 0
        features["num_carring"] = 0
        features["dead_end"] = 0
        features["closest_enemy_pacman"] *= 40
        features["closest_enemy_pacman_vertical"] *= 40
        
      elif self.mode == "ATTACK":
        features["defence"] = 0
        features["defence_ancor_point"] = 0
        features["reverse"] = 0

      # when it is timeout the following features are not important
      elif self.mode == "TIMEOUT":
        features["closest-food"] = 0
        features["num_carring"] = 0
        features["dead_end"] = 0
        features["closest_enemy_pacman"] = 0
        features["closest_enemy_pacman_vertical"]  = 0
        features["defence"] = 0
        features["defence_ancor_point"] = 0
        features["reverse"] = 0
        features["home_anchor_point"] = closest_way_home / self.max_distance
         
    features["defence_ancor_point"] = self.get_maze_distance((next_x, next_y), (self.defence_ancor_point))
    return features

  def get_maze_distance(self, pos1, pos2):
    """
    Returns the distance between pos1 and pos2 using A* search.
    Cost is 1.0 for own territory and 1.2 for enemy territory.
    """
    # Ensure coordinates are integers
    pos1 = (int(pos1[0]), int(pos1[1]))
    pos2 = (int(pos2[0]), int(pos2[1]))

    if pos1 == pos2:
        return 0

    # Priority Queue for A*: stores ((x, y), cost) ordered by f = cost + heuristic
    pq = util.PriorityQueue()
    pq.push((pos1, 0), 0)
    
    # Keep track of lowest cost to reach a node
    visited = {} # format: { (x,y): cost }
    
    mid_width = self.layout_width // 2

    while not pq.is_empty():
        current_pos, current_g = pq.pop()

        if current_pos == pos2:
            return current_g

        # Optimization: if we found a shorter way to this node already, skip
        if current_pos in visited and visited[current_pos] <= current_g:
            continue
        visited[current_pos] = current_g

        # Expand neighbors
        neighbors = Actions.get_legal_neighbors(current_pos, self.walls)
        for next_pos in neighbors:
            nx, ny = next_pos
            
            # Calculate cost to move TO next_pos
            step_cost = self.own_cost
            
            # Check territory based on team color
            is_enemy_territory = False
            if self.red:
                # Red team: Left is home, Right (>= mid) is enemy
                if nx >= mid_width: is_enemy_territory = True
            else:
                # Blue team: Right is home, Left (< mid) is enemy
                if nx < mid_width: is_enemy_territory = True
            
            if is_enemy_territory:
                step_cost = self.enemy_cost

            new_g = current_g + step_cost

            # Check if this path is better than any previously found path to next_pos
            if next_pos not in visited or new_g < visited[next_pos]:
                # Manhattan distance heuristic
                h = abs(nx - pos2[0]) + abs(ny - pos2[1])
                f = new_g + h
                pq.push((next_pos, new_g), f)

    # Return a large number if no path found (shouldn't happen in standard mazes unless enclosed)
    return 999999
  
  
  
  def get_weights(self):
    # [ defence weights, attack weights]
    return [
      {
        "closest-food": -1500,
        "closest_enemy_ghost": -300,
        "closest_enemy_pacman": 2000,
        "closest_enemy_pacman_vertical": 500,
        # "food_left": 0,
        # "score": 0,
        "closest_capsule": 0,
        "num_carring": -100,
        "dead_end": -100,
        "defence": 500,
        "defence_ancor_point": -100,
        "stop": -150,
        "reverse": -50,
        "dead": 10000,
        "home_anchor_point": 0
      },
      {
        "closest-food": -2500,
        "closest_enemy_ghost": -300,
        "closest_enemy_pacman": 100,
        "closest_enemy_pacman_vertical": 50,
        # "food_left": 0,         
        # "score": 0,
        "closest_capsule": -40,
        "num_carring": -3500,
        "dead_end": -350,
        "defence": 500,
        "defence_ancor_point": 0,
        "stop": -50,
        "reverse": -5,
        "dead": 100000,
        "home_anchor_point": -2000 
      }
    ]
  def initial_weight(self):
    teammate_index = (self.index + 2) % 4
    # 0 defend 1 atack
    # let the smaller number be the attack
    weights_index = 1 if self.index < teammate_index else 0
    # print("Agent index:", self.index, "Weights index:", weights_index)
    self.weights = self.weights_array[weights_index]
    self.current_mode = "ATTACK" if self.index < teammate_index else "DEFENCE"

# LIGHTBLUE
# DEFENCIVE
class Agent1(DDagent):
  def delete():
    pass

class Agent2(DDagent):
  def delete():
    pass
  
  
#Agent index: 1 Weights index: 1 ATTACK
#Agent index: 3 Weights index: 0 DEFENCE



# Defensive Checkpoints for agent 1: (18, 3) and (18, 11)
# Default Defence Ancor Points: (17, 14) (17, 3)