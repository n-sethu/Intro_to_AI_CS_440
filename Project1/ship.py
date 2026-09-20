# Grid generation
from enum import IntEnum
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# magic number enum class
class CellState(IntEnum):
    BLOCKED=0
    OPEN=1
    FIRE=2

def grid_generation(D:int)->np.ndarray:
    # small size(1 byte) of each cell so we can use larger D
    layout= np.full((D,D),CellState.BLOCKED,dtype=np.int8)
    # question - do we include the edges?
    x = rd.randint(1,D-2)
    y = rd.randint(1,D-2)
    layout[x,y]=CellState.OPEN
    
    # First step: blocked cells with one open neighbot
    L, coords = get_L(layout)
    while(L>0):
        to_open = rd.randint(0,L-1)
        r, c = coords[to_open]
        
        layout[r, c] = CellState.OPEN
        L, coords = get_L(layout)
    
    
    # Dead end step
    F, F_coords = get_F(layout)
    goal = F//2
    while(F>goal):
        chosen_f = rd.randint(0,F-1)
        F_r, F_c = F_coords[chosen_f]
        closed_neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = F_r + dx, F_c + dy

            # Checking boundaries
            if 0 <= nr < D and 0 <= nc < D and layout[nr, nc] == CellState.BLOCKED:
                closed_neighbors.append((nr, nc))
        
        if closed_neighbors:
            nr, nc = rd.choice(closed_neighbors)
            layout[nr, nc] = CellState.OPEN
            
        F, F_coords = get_F(layout)
        
    return layout
        
        
    
    

# Let L be the number of currently blocked cells that have exactly one open neighbor.
def get_L(layout:np.ndarray):
    blocked = (layout==CellState.BLOCKED)
    open = (layout==CellState.OPEN)
    
    open_neighbors = np.zeros_like(layout, dtype=np.int8)

    
    open_neighbors[:-1, :] += open[1:, :]   # Look Down
    open_neighbors[1:, :]  += open[:-1, :]   # Look Up
    open_neighbors[:, :-1] += open[:, 1:]   # Look Right
    open_neighbors[:, 1:]  += open[:, :-1]   # Look Left
    
    matched = blocked & (open_neighbors==1)
    
    count = int(np.sum(matched))
    
    coords = np.argwhere(matched)
    
    return count, coords

# Let F be the number of dead ends
def get_F(layout):
    open = (layout==CellState.OPEN)
    
    open_neighbors = np.zeros_like(layout, dtype=np.int8)

    open_neighbors[:-1, :] += open[1:, :]   # Look Down
    open_neighbors[1:, :]  += open[:-1, :]   # Look Up
    open_neighbors[:, :-1] += open[:, 1:]   # Look Right
    open_neighbors[:, 1:]  += open[:, :-1]   # Look Left
    
    matched = open & (open_neighbors==1)
    
    count = int(np.sum(matched))
    
    coords = np.argwhere(matched)
    
    return count, coords
    
layout = grid_generation(10)   
print(layout)
plt.imshow(layout, cmap='viridis') 
plt.colorbar() 
plt.title("2D Array Visualization (imshow)")
plt.show()