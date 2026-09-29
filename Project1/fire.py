#fire spread
import random

class FireSystem:

    def __init__(self, ship, q: float, start_cell: tuple):
        self.ship = ship
        self.q = q #flamability 
        self.burning_cells = {start_cell}  
        #set of (x,y) tuples that are burning (O(1) lookup for if a cell is burning)

    def is_burning(self, cell) -> bool:
        return cell in self.burning_cells

    def step(self):
        #for each cell that is next to a fire (K > 0), count the neighbors that are on fire
        k_count = {}
        for cell in self.burning_cells:
            for nb in self.ship.get_open_neighbors(cell):
                if nb not in self.burning_cells:
                    k_count[nb] = k_count.get(nb, 0) + 1

        #decide which vulnerable cells will catch on fire based on the ignition probability
        newly_lit = set()
        for cell, k in k_count.items():
            if random.random() < 1 - (1 - self.q) ** k:
                newly_lit.add(cell)
        #spread the fire
        self.burning_cells |= newly_lit

    #the set of open cells that are adjacent to a fire cell
    def neighbors_of_fire(self) -> set:
        result = set()
        for cell in self.burning_cells:
            result.update(self.ship.get_open_neighbors(cell))
        return result - self.burning_cells

#to run just fire.py
if __name__ == "__main__":
    from ship import ShipGrid
    ship = ShipGrid(30) #small grid to test
    start = random.choice(ship.get_open_cells()) #randomly start the fire at an open cell
    fire = FireSystem(ship, q=0.3, start_cell=start)
    for t in range(1, 11): #test length of time
        fire.step()
        print(f"t={t}: {len(fire.burning_cells)} burning")

    #check that a wall cell is not on fire
    assert all(ship.is_open(r, c) for r, c in fire.burning_cells), "fire in a wall!"
    print("fire only in open cells: True")