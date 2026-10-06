# bot classes
# @zahra can you make 4 different classes that inherit a base bot classes, lmk if this is too much to finish with the
# pathfinding algorithms (idt bots 1-3 shld be too much tho)
from pathfinding import Pathfinder

class BaseBot: #Base bot that all the other bots inherit

    def __init__(self, ship, start, button, initial_fire):
        self.ship = ship
        self.position = start
        self.button = button
        self.initial_fire = initial_fire

    #every bot must have its own funtion to choose the next move
    def choose_move(self, fire):
        raise NotImplementedError

    def first_step(self, path): #takes the path returned from dfs and gets the first step
        if path is None or len(path) < 2:
            return self.position
        return path[1]

class Bot1(BaseBot):
    #Bot1 plans its path once, avoiding the initial fire cell
    def __init__(self, ship, start, button, initial_fire):
        super().__init__(ship, start, button, initial_fire)
        path = Pathfinder.next_move_astar(ship, start, button, {initial_fire})
        # path = Pathfinder.next_move_bfs(ship, start, button, {initial_fire})

        self.path = path if path is not None else [start]
        self.indx = 0

    def choose_move(self, fire):
        if self.indx + 1 < len(self.path):
            self.indx += 1
            self.position = self.path[self.indx]
        return self.position


class Bot2(BaseBot):
    #Bot2 re plans its path at every step, avoiding all buring cells
    def choose_move(self, fire):
        blocked = set(fire.burning_cells)
        # path = Pathfinder.next_move_bfs(self.ship, self.position, self.button, blocked)
        path = Pathfinder.next_move_astar(self.ship, self.position, self.button, blocked)

        self.position = self.first_step(path)
        return self.position

class Bot3(BaseBot):
    #Bot3 re plans its path at every step, avoiding all buring cells + cells adjacent to burning cells if possible
    def choose_move(self, fire):
        burning = set(fire.burning_cells)
        cautious = burning | fire.neighbors_of_fire()
        path = Pathfinder.next_move_astar(self.ship, self.position, self.button, cautious)
        # path = Pathfinder.next_move_bfs(self.ship, self.position, self.button, cautious)

        if path is None:                   #not possible so revert to Bot2 behavior
            # path = Pathfinder.next_move_bfs(self.ship, self.position, self.button, burning)
            path = Pathfinder.next_move_astar(self.ship, self.position, self.button, burning)

        self.position = self.first_step(path)
        return self.position

class Bot4(BaseBot):
    """Plan through predicted fire risk in space and time."""

    def __init__(
        self,
        ship,
        start,
        button,
        initial_fire,
        replanning_interval=1,
        risk_threshold=0.4,
        risk_weight=5.0,
        heuristic_weight=1.0,
    ):
        super().__init__(ship, start, button, initial_fire)
        self.replanning_interval = max(1, replanning_interval)
        self.risk_threshold = risk_threshold
        self.risk_weight = risk_weight
        self.heuristic_weight = heuristic_weight
        self.steps_since_plan = self.replanning_interval
        self.path = None
        self.path_index = 0

    def _plan(self, fire):
        burning = set(fire.burning_cells)
        shortest_path = Pathfinder.next_move_astar(
            self.ship, self.position, self.button, burning
        )
        self.path = None
        self.path_index = 0
        self.steps_since_plan = 0
        if shortest_path is None:
            return
        # Maze routes can be much longer than their Manhattan distance.
        distance = len(shortest_path) - 1
        horizon = max(distance + 10, 2 * distance)
        risk = Pathfinder.predict_fire_risk(
            self.ship, fire.burning_cells, fire.q, horizon
        )

        # Relax the hard cutoff only when the preferred safety bound has no path.
        thresholds = sorted({self.risk_threshold, 0.6, 0.8, 0.99})
        for threshold in thresholds:
            path = Pathfinder.next_move_risk_astar(
                self.ship,
                self.position,
                self.button,
                risk,
                threshold,
                self.risk_weight,
                self.heuristic_weight,
                blocked_set=burning,
            )
            if path is not None:
                self.path = path
                self.path_index = 0
                self.steps_since_plan = 0
                return

        # If every predicted route exceeds the cutoff, take a currently clear
        # escape route rather than wait for the fire to reach us.
        self.path = shortest_path

    def choose_move(self, fire):
        # Replan from the observed fire after each configured batch of moves.
        stale_path = self.path is not None and any(
            cell in fire.burning_cells
            for cell in self.path[self.path_index + 1:]
        )
        if (self.path is None or stale_path
                or self.steps_since_plan >= self.replanning_interval):
            self._plan(fire)

        if self.path is not None and self.path_index + 1 < len(self.path):
            next_cell = self.path[self.path_index + 1]
            if next_cell not in fire.burning_cells:
                self.path_index += 1
                self.position = next_cell

        self.steps_since_plan += 1
        return self.position
