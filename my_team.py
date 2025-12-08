from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point
import contest.util as util
import random


def create_team(first_index, second_index, is_red,
                first='ComplexStateAgent', second='ComplexStateAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class ComplexStateAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start_pos = game_state.get_agent_position(self.index)
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        # Identity
        self.red = game_state.is_on_red_team(self.index)
        self.teammate_index = (self.index + 2) % 4
        
        # Map Analysis
        self.mid_x = game_state.data.layout.width // 2
        if self.red: self.mid_x -= 1
        self.boundary_points = self.get_boundary_points(game_state)
        self.safe_areas = [pos for group in self.boundary_points for pos in group]
        
        # State Variables
        self.current_state = 'ATTACK_FOOD'
        self.target_position = None
        
            # Initialize state maps
        self.compute_dead_ends(game_state)
        self.depth_binary_mask = [[1 if self.dead_ends_depth[y][x] > 0 else 0 for x in range(self.width)]for y in range(self.height)]

    def get_boundary_points(self, game_state):
        points = []
        group = []
        prev_wall = False
        for y in range(self.height):
            curr_wall = game_state.has_wall(self.mid_x, y)
            if not curr_wall:
                if prev_wall and len(group) > 0:
                    points.append(group)
                    group = []
                group.append((self.mid_x,y))
            prev_wall = curr_wall
        if len(group)> 0: points.append(group)
        return points

    def compute_dead_ends(self, game_state):
        walls = game_state.get_walls()
        W, H = walls.width, walls.height

        # degree[y][x] = number of free neighbor CELLS (no STOP)
        degree = [[0 for _ in range(W)] for _ in range(H)]
        temp = [[0 for _ in range(W)] for _ in range(H)]
        visited = [[False for _ in range(W)] for _ in range(H)]

        # 1) compute degrees using neighbor cells
        for x in range(W):
            for y in range(H):
                if walls[x][y]:
                    continue
                neighbors = Actions.get_legal_neighbors((x, y), walls)  # returns list of (nx, ny)
                degree[y][x] = len(neighbors) - 1  # 0..4, STOP excluded

        # 2) start BFS from true dead-ends (degree == 1)
        queue = util.Queue()
        for y in range(H):
            for x in range(W):
                if degree[y][x] == 1:
                    temp[y][x] = 1      # distance-from-leaf
                    visited[y][x] = True
                    queue.push((x, y))

        # 3) BFS outward through corridors (degree == 2)
        while not queue.is_empty():
            x, y = queue.pop()
            for nx, ny in Actions.get_legal_neighbors((x, y), walls):
                if visited[ny][nx]:
                    continue
                # only propagate through corridor cells (degree 2)
                if degree[ny][nx] == 2:
                    visited[ny][nx] = True
                    temp[ny][nx] = temp[y][x] + 1
                    queue.push((nx, ny))

        # 4) Now temp contains distance-from-closest-leaf (leaf=1, increases towards junction).
        #    We need to reverse numbering inside each connected component of temp>0
        final = [[0 for _ in range(W)] for _ in range(H)]
        comp_visited = [[False for _ in range(W)] for _ in range(H)]

        for y in range(H):
            for x in range(W):
                if temp[y][x] > 0 and not comp_visited[y][x]:
                    # flood-fill this component of temp>0 cells
                    stack = [(x, y)]
                    comp_cells = []
                    comp_max = 0
                    comp_visited[y][x] = True

                    while stack:
                        cx, cy = stack.pop()
                        comp_cells.append((cx, cy))
                        if temp[cy][cx] > comp_max:
                            comp_max = temp[cy][cx]

                        for nx, ny in Actions.get_legal_neighbors((cx, cy), walls):
                            if temp[ny][nx] > 0 and not comp_visited[ny][nx]:
                                comp_visited[ny][nx] = True
                                stack.append((nx, ny))

                    # reverse depths inside this component: entrance=1 ... deepest=comp_max
                    for cx, cy in comp_cells:
                        final[cy][cx] = comp_max - temp[cy][cx] + 1

        # store into self if you want
        self.dead_ends_depth = final
        return final

    def get_non_deadend_actions(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        legal_actions = []
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            nx,ny = successor.get_agent_state(self.index).get_position()
            nx,ny = int(nx), int(ny)
            if self.depth_binary_mask[ny][nx] == 0:
                legal_actions.append(action)
        return legal_actions
    
    def update_state(self, game_state, obs):
        # If a ghost is within 5 units, switch to CONSCIOUS mode
        dist_home = min([self.get_maze_distance(obs['my_pos'],p) for p in self.safe_areas])
        num_carring = game_state.get_agent_state(self.index).num_carrying
        if len(obs['both_agent_food']) <= 2:
            if self.current_state != "HOME":
                print(self.index, self.current_state,"HOME")
            self.current_state = "HOME"
        elif num_carring > 0 and dist_home + 1 >= game_state.data.timeleft / 4.:
            if self.current_state != "HOME":
                print(self.index, self.current_state,"HOME TIMEOUT")
            self.current_state = "HOME"
        elif obs['closest_defender_dist'] <= 4:
            self.current_state = 'CONSCIOUS_ATTACK_FOOD'
            if self.current_state != "CONSCIOUS_ATTACK_FOOD":
                print(self.index, self.current_state,"CONSCIOUS_ATTACK_FOOD")
        else:
            if self.current_state != "ATTACK_FOOD":
                print(self.index, self.current_state,"ATTACK_FOOD")
            self.current_state = 'ATTACK_FOOD'
            
            
    def observe(self, game_state):
        obs = {}
        my_pos = game_state.get_agent_position(self.index)
        obs['my_pos'] = my_pos
        
        # Enemies
        opponents = self.get_opponents(game_state)
        obs['enemies'] = [game_state.get_agent_state(i) for i in opponents]
        
        # Defenders (Enemy Ghosts in their territory)
        obs['defenders'] = [e for e in obs['enemies'] if not e.is_pacman and e.get_position() and e.scared_timer < 5]
        
        # Food
        foods = self.get_food(game_state).as_list()
        obs['both_agent_food'] = foods
        # avg_height = sum([ y for (x, y) in foods])/len(foods)
        # print(avg_height,self.height//2)
        # Split food logic
        if self.index < self.teammate_index:
            obs['food'] = [ (x, y) for (x, y) in foods if y >= self.height // 2]
            if not obs['food']: obs['food'] = foods
        else:
            obs['food'] = [ (x, y) for (x, y) in foods if y < self.height // 2]
            if not obs['food']: obs['food'] = foods
        # print(obs['food'])
        # Distances
        obs['dist_home'] = min([self.get_maze_distance(my_pos, pos) for pos in self.safe_areas])

        if len(obs['defenders']) > 0:
            obs['closest_defender_dist'] = min([self.get_maze_distance(my_pos, d.get_position()) for d in obs['defenders']])
        else:
            obs['closest_defender_dist'] = 999
            
        # Dangerous Cells (Ghost Exclusion Zones)
        # We treat the ghost's position AND the cells immediately around it as walls
        danger = set()
        for d in obs['defenders']:
            p = d.get_position()
            if p:
                px, py = int(p[0]), int(p[1])
                danger.add((px, py))
                # Add radius 1 buffer (Standard Pacman kill zone)
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    danger.add((px+dx, py+dy))
                    
        # Add dead ends to danger if a ghost is chasing us? 
        # Optional: You could add deep dead-ends to danger here, but A* handles it naturally.
        obs['dangerous_cells'] = danger
        
        return obs

    def find_escape_action(self, game_state, start_pos, unsafe_positions):
        """
        Uses A* to find the shortest path to home that strictly avoids unsafe positions.
        """
        # Priority Queue Item: (priority, current_position, path_of_actions)
        queue = util.PriorityQueue()
        queue.push((start_pos, []), 0)
        
        visited = set()
        visited.add(start_pos)
        
        # Optimization: Simple heuristic is Manhattan distance to the midline
        # Because 'home' is a long vertical line, not a single point.
        
        while not queue.is_empty():
            curr_pos, path = queue.pop()
            
            # If we reached any safe area, return the first step of the path
            if curr_pos in self.safe_areas:
                if len(path) > 0:
                    return path[0]
                return None # Already home

            # Performance Cutoff (prevent timeout on huge mazes)
            if len(path) > 30: 
                continue 

            x, y = int(curr_pos[0]), int(curr_pos[1])
            
            # Get legal neighbors (pass walls to save time)
            neighbors = Actions.get_legal_neighbors((x, y), game_state.get_walls())
            
            for nx, ny in neighbors:
                next_pos = (nx, ny)
                
                # Skip visited or dangerous cells
                if next_pos in visited: continue
                if next_pos in unsafe_positions: continue
                
                visited.add(next_pos)
                
                # Determine Direction
                dx, dy = nx - x, ny - y
                if dx == 1: action = 'East'
                elif dx == -1: action = 'West'
                elif dy == 1: action = 'North'
                elif dy == -1: action = 'South'
                else: action = 'Stop'
                
                new_path = path + [action]
                
                # Heuristic: Manhattan Distance to the safe x-boundary (self.mid_x)
                # This encourages moving toward the safe side.
                h_score = abs(nx - self.mid_x)
                g_score = len(new_path)
                f_score = g_score + h_score
                
                queue.push((next_pos, new_path), f_score)
                
        return None # No path found (Trapped)

    def choose_action(self, game_state):
        obs = self.observe(game_state)
        self.update_state(game_state, obs)
        
        my_pos = obs['my_pos']
        actions = game_state.get_legal_actions(self.index)
        non_deadend_action = self.get_non_deadend_actions(game_state)

        # --- MODE: HOME (ESCAPE) ---
        if self.current_state == "HOME":
            # 1. Try to find a clean path home using A*
            best_action = self.find_escape_action(game_state, my_pos, obs['dangerous_cells'])
            
            if best_action:
                return best_action
                
            # 2. FALLBACK: SURVIVAL MODE
            # If A* returns None, we are trapped or blocked. 
            # We must just maximize distance to the closest ghost to stall.
            best_survival_action = None
            max_dist = -1
            
            # Filter moves that kill us immediately
            safe_moves = []
            for action in actions:
                successor = game_state.generate_successor(self.index, action)
                next_pos = successor.get_agent_state(self.index).get_position()
                if next_pos not in obs['dangerous_cells']:
                    safe_moves.append((action, next_pos))
            
            # If no safe moves, just pick random legal and pray
            if not safe_moves:
                return random.choice(actions)
                
            # Pick the move that puts us furthest from the closest defender
            for action, next_pos in safe_moves:
                # Calculate dist to closest defender
                if obs['defenders']:
                    dist = min([self.get_maze_distance(next_pos, d.get_position()) for d in obs['defenders']])
                else:
                    dist = 0
                
                if dist > max_dist:
                    max_dist = dist
                    best_survival_action = action
            
            return best_survival_action if best_survival_action else random.choice(actions)

        # --- MODE 1: AGGRESSIVE SWEEP ---
        if self.current_state == 'ATTACK_FOOD':
            # ... (Keep your existing logic here) ...
            pass 
            # (Adding it back briefly so the function is complete for context)
            min_dist_to_food = 999999
            best_a = random.choice(actions)
            for food in obs['food']:
                dist = self.get_maze_distance(my_pos, food)
                if dist < min_dist_to_food:
                    min_dist_to_food = dist
            for action in actions:
                successor = game_state.generate_successor(self.index, action)
                new_pos = successor.get_agent_state(self.index).get_position()
                new_min = min([self.get_maze_distance(new_pos,f) for f in obs['food']]) if obs['food'] else 0
                if min_dist_to_food > new_min:
                    return action
            return best_a

        # --- MODE 2: CONSCIOUS ATTACK ---
        elif self.current_state == 'CONSCIOUS_ATTACK_FOOD':
            # This logic was mostly fine, just ensuring it returns something
            x,y = int(my_pos[0]), int(my_pos[1])

            # If inside a dead end, only move OUT (deeper < shallower)
            if self.depth_binary_mask[y][x] == 1:
                best_depth_action = None
                curr_depth = self.dead_ends_depth[y][x]
                for action in actions:
                    successor = game_state.generate_successor(self.index, action)
                    nx,ny = successor.get_agent_state(self.index).get_position()
                    nx,ny = int(nx), int(ny)
                    # We want to go to a cell with LOWER depth value (closer to exit)
                    if self.dead_ends_depth[ny][nx] < curr_depth: 
                        return action
                return random.choice(actions) # Should not happen in valid dead end

            else:
                # Not in a dead end: Avoid entering one if ghost is near
                legal_food = [f for f in obs['food'] if self.depth_binary_mask[f[1]][f[0]] == 0]
                if not legal_food: legal_food = obs['food']

                min_enemy = min([self.get_maze_distance(my_pos,g.get_position()) for g in obs['defenders']])
                min_food = min([self.get_maze_distance(my_pos,f) for f in legal_food]) if legal_food else 0
                
                best_action = None
                # Score actions: +EnemyDist, -FoodDist
                best_score = -99999

                for action in non_deadend_action:
                    successor = game_state.generate_successor(self.index, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    
                    # Avoid dangerous cells strictly here too
                    if new_pos in obs['dangerous_cells']: continue

                    d_food = min([self.get_maze_distance(new_pos,f) for f in legal_food]) if legal_food else 0
                    d_enemy = min([self.get_maze_distance(new_pos,f.get_position()) for f in obs['defenders']])
                    
                    # Heuristic score
                    score = d_enemy * 2 - d_food 
                    if score > best_score:
                        best_score = score
                        best_action = action
                
                if best_action: return best_action
                return random.choice(non_deadend_action) if non_deadend_action else random.choice(actions)

        return random.choice(actions)
