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
        path = Pathfinder.next_move_bfs(ship, start, button, {initial_fire})
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
        path = Pathfinder.next_move_bfs(self.ship, self.position, self.button, blocked)
        self.position = self.first_step(path)
        return self.position

class Bot3(BaseBot):
    #Bot3 re plans its path at every step, avoiding all buring cells + cells adjacent to burning cells if possible
    def choose_move(self, fire):
        burning = set(fire.burning_cells)
        cautious = burning | fire.neighbors_of_fire()
        path = Pathfinder.next_move_bfs(self.ship, self.position, self.button, cautious)
        if path is None:                   #not possible so revert to Bot2 behavior
            path = Pathfinder.next_move_bfs(self.ship, self.position, self.button, burning)
        self.position = self.first_step(path)
        return self.position

#bot4 next