from collections import deque
# A*, bfs, etc...
    
# idea: keep the list of moves in a tuple/list, store each move taken in a separate one,
# for bot 1 just iterate through list and check if (r,c) is burning at that timestep
# bot 2 - recalculate from the new startPos each time (checks if each (r,c) is adjacent to a burning cell)
# bot 3: edit the risk parameter
# @zahra -> can you do A* or the rest of this file

class Pathfinder:
    
    @staticmethod
    def next_move_bfs(ship_grid, start: tuple, goal: tuple, blocked_set:set):
        if start == goal:
            return start
        queue = deque([start])
        parent = {start:None}
        
        # reconstruct moves
        while queue:
            curr = queue.popleft()
            
            if curr == goal:
                path = []
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path[3]
            for neighbor in ship_grid.get_open_neighbors(curr):
                if neighbor not in parent and neighbor not in blocked_set:
                    parent[neighbor]= curr
                    queue.append(neighbor)
            
        return None