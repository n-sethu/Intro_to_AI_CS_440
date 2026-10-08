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
    def predict_fire_risk(ship_grid, burning_cells: set, q: float, horizon: int):
        """Estimate the probability that each open cell is burning at each time."""
        # Each layer is the predicted fire state at one future timestep.
        risk = [
            [[0.0 for _ in range(ship_grid.D)] for _ in range(ship_grid.D)]
            for _ in range(horizon + 1)
        ]

        for row, col in burning_cells:
            risk[0][row][col] = 1.0

        open_cells = ship_grid.get_open_cells()
        neighbors = {cell: ship_grid.get_open_neighbors(cell) for cell in open_cells}
        # Propagate risk using the independent-neighbor ignition approximation.
        for time in range(horizon):
            for row, col in open_cells:
                current_risk = risk[time][row][col]
                if current_risk >= 1.0:
                    risk[time + 1][row][col] = 1.0
                    continue

                no_ignition = 1.0
                for neighbor in neighbors[(row, col)]:
                    neighbor_risk = risk[time][neighbor[0]][neighbor[1]]
                    no_ignition *= 1.0 - q * neighbor_risk

                ignition_risk = 1.0 - no_ignition
                risk[time + 1][row][col] = (
                    current_risk + (1.0 - current_risk) * ignition_risk
                )

        return risk

    @staticmethod
    def next_move_risk_astar(
        ship_grid,
        start: tuple[int, int],
        goal: tuple[int, int],
        risk: list[list[list[float]]],
        risk_threshold: float,
        risk_weight: float,
        heuristic_weight: float = 1.0,
        blocked_set: set | None = None,
    ):
        """Find a path through (row, col, time) states using predicted risk."""
        blocked_set = blocked_set or set()
        if start in blocked_set or goal in blocked_set:
            return None
        if start == goal:
            return [start]

        blocked = blocked_set if blocked_set is not None else set()
        horizon = len(risk) - 1

        def heuristic(cell):
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

        # Time is part of the state because the same cell can have different risk later.
        start_state: tuple[int, int, int] = (start[0], start[1], 0)
        counter = 0
        open_set: list[tuple[float, int, int, tuple[int, int, int]]] = [
            (heuristic_weight * heuristic(start), heuristic(start), counter, start_state)
        ]
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {
            start_state: None
        }
        g_score: dict[tuple[int, int, int], float] = {start_state: 0.0}
        # Keep tradeoffs: an earlier arrival only dominates if it costs no more.
        arrivals = {start: [(0, 0.0)]}

        while open_set:
            _, _, _, current = heappop(open_set)
            current_cost = g_score[current]
            row, col, time = current
            cell = (row, col)
            # Skip queued entries superseded by cheaper or earlier arrivals.
            if (time, current_cost) not in arrivals.get(cell, []):
                continue
            if time + heuristic(cell) > horizon:
                continue

            if (row, col) == goal:
                path = []
                while current is not None:
                    path.append((current[0], current[1]))
                    current = parent[current]
                path.reverse()
                return path

            if time >= horizon:
                continue

            # Fire only grows: waiting cannot make a route safer.
            next_cells = ship_grid.get_open_neighbors((row, col))
            for next_row, next_col in next_cells:
                if (next_row, next_col) in blocked:
                    continue
                next_time = time + 1
                next_heuristic = heuristic((next_row, next_col))
                # Even an obstacle-free route must fit within the forecast.
                if next_time + next_heuristic > horizon:
                    continue
                # Reaching the button ends the trial before fire spreads.
                risk_time = time if (next_row, next_col) == goal else next_time
                next_risk = risk[risk_time][next_row][next_col]
                # Reject unsafe arrivals before applying the softer risk cost.
                if (risk[0][next_row][next_col] >= 1.0
                        or next_risk >= 1.0 or next_risk > risk_threshold):
                    continue

                next_state = (next_row, next_col, next_time)
                tentative_cost = current_cost + 1.0 + risk_weight * next_risk
                if tentative_cost >= g_score.get(next_state, float("inf")):
                    continue

                cell = (next_row, next_col)
                known = arrivals.get(cell, [])
                if any(t <= next_time and cost <= tentative_cost for t, cost in known):
                    continue
                arrivals[cell] = [
                    (t, cost) for t, cost in known
                    if not (next_time <= t and tentative_cost <= cost)
                ]
                arrivals[cell].append((next_time, tentative_cost))

                g_score[next_state] = tentative_cost
                parent[next_state] = current
                counter += 1
                priority = tentative_cost + heuristic_weight * next_heuristic
                heappush(open_set, (priority, next_heuristic, counter, next_state))

        return None

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
