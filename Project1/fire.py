# fire spread
class FireSystem:
    
    def __init__(self, q:float):
        self.q = q
        self.burning_cells = set() 
        # set of (x,y) tuples that are burning (O(1) lookup for if a cell is burning)
        
        