from collections import deque
from heapq import heappop, heappush
# A*, bfs, etc...
    
# idea: keep the list of moves in a tuple/list, store each move taken in a separate one,
# for bot 1 just iterate through list and check if (r,c) is burning at that timestep
# bot 2 - recalculate from the new startPos each time (checks if each (r,c) is adjacent to a burning cell)
# bot 3: edit the risk parameter
# @zahra -> can you do A* or the rest of this file

class Pathfinder:

    @staticmethod
    def next_move_astar(ship_grid, start: tuple, goal: tuple, blocked_set: set):
        if start == goal:
            return [start]

        def heuristic(cell):
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

        open_set = [(heuristic(start), 0, start)]
        parent: dict[tuple, tuple | None] = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current_cost, current = heappop(open_set)

            if current_cost != g_score[current]:
                continue

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path

            for neighbor in ship_grid.get_open_neighbors(current):
                if neighbor in blocked_set:
                    continue

                tentative_cost = current_cost + 1
                if tentative_cost < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_cost
                    parent[neighbor] = current
                    priority = tentative_cost + heuristic(neighbor)
                    heappush(open_set, (priority, tentative_cost, neighbor))

        return None
    
    @staticmethod
    def next_move_bfs(ship_grid, start: tuple, goal: tuple, blocked_set:set):
        if start == goal:
            return [start]
        queue = deque([start])
        parent: dict[tuple, tuple | None] = {start:None}
        
        # reconstruct moves
        while queue:
            curr = queue.popleft()
            
            if curr == goal:
                path = []
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path
            for neighbor in ship_grid.get_open_neighbors(curr):
                if neighbor not in parent and neighbor not in blocked_set:
                    parent[neighbor]= curr
                    queue.append(neighbor)
            
        return None