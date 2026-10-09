# regression check for bot 4's fire avoidance and simulation timing
import unittest

from bots import Bot4
from fire import FireSystem
from pathfinding import Pathfinder

class SmallGrid:
    def __init__(self, size, cells):
        self.D = size
        self.cells = set(cells)

    def get_open_cells(self):
        return sorted(self.cells)

    def get_open_neighbors(self, cell):
        row, col = cell
        return [(row + dr, col + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (row + dr, col + dc) in self.cells]

class Bot4Tests(unittest.TestCase):
    def test_relaxed_threshold_never_crosses_existing_fire(self):
        ship = SmallGrid(3, {(1, 0), (1, 1), (1, 2)})
        fire = FireSystem(ship, 0.0, (1, 1))
        risk = Pathfinder.predict_fire_risk(ship, fire.burning_cells, 0.0, 4)
        self.assertIsNone(Pathfinder.next_move_risk_astar(
            ship, (1, 0), (1, 2), risk, 1.0, 5.0))
        bot = Bot4(ship, (1, 0), (1, 2), (1, 1))
        self.assertEqual(bot.choose_move(fire), (1, 0))

    def test_cached_path_is_invalidated_when_fire_spreads(self):
        ship = SmallGrid(3, {(r, c) for r in range(3) for c in range(3)})
        fire = FireSystem(ship, 0.0, (2, 2))
        bot = Bot4(ship, (0, 0), (0, 2), (2, 2), replanning_interval=3)
        self.assertEqual(bot.choose_move(fire), (0, 1))
        fire.burning_cells.add((0, 2))
        self.assertNotIn(bot.choose_move(fire), fire.burning_cells)
        self.assertIsNone(bot.path)

    def test_button_is_reached_before_fire_spreads(self):
        ship = SmallGrid(3, {(1, 0), (1, 1), (1, 2)})
        fire = FireSystem(ship, 1.0, (1, 2))
        risk = Pathfinder.predict_fire_risk(ship, fire.burning_cells, 1.0, 2)
        self.assertEqual(Pathfinder.next_move_risk_astar(
            ship, (1, 0), (1, 1), risk, 0.4, 5.0,
            blocked_set=fire.burning_cells), [(1, 0), (1, 1)])

    def test_horizon_uses_maze_distance(self):
        cells = {(0, c) for c in range(7)} | {(6, c) for c in range(7)}
        cells |= {(r, 6) for r in range(7)}
        cells |= {(r, 0) for r in range(2, 7)}
        cells.add((3, 3))
        ship = SmallGrid(7, cells)
        fire = FireSystem(ship, 0.0, (3, 3))
        bot = Bot4(ship, (0, 0), (2, 0), (3, 3))
        bot._plan(fire)
        self.assertEqual(len(bot.path), 23)
        self.assertTrue(all(cell not in fire.burning_cells for cell in bot.path))

if __name__ == "__main__":
    unittest.main()
