from collections import deque
from enum import Enum
import random

import numpy as np


class DetectorResult(Enum):
    NO_PING = 0
    PING = 1
    FOUND = 2


class LocatorBot:
# Shared methods for location
    def __init__(self, ship, start, lostBot):
        self.ship = ship
        self.position = tuple(start)
        self._lost_bot = tuple(lostBot)
        self.open_cells = {tuple(cell) for cell in ship.get_open_cells()}
        if self.position not in self.open_cells or self._lost_bot not in self.open_cells:
            raise ValueError("Both bots must start on open cells.")
        self.found = False
        self.move_actions = 0
        self.detector_actions = 0

    @property
    def cost(self):
        return self.move_actions + 0.5 * self.detector_actions

    def _detect(self):
        # ping or no ping
        raise NotImplementedError

    def ping(self):
        # runs a scan, stops when its on the actual bot
        if self.found:
            return DetectorResult.FOUND
        self.detector_actions += 1
        if self.position == self._lost_bot:
            self.found = True
            return DetectorResult.FOUND
        return self._detect()

    def move(self, destination):
        # takes step and adds to action counter
        if self.found:
            raise RuntimeError("The search has already finished.")
        destination = tuple(destination)
        if destination not in self.ship.get_open_neighbors(self.position):
            raise ValueError("Destination must be an open neighbor of the current cell.")
        self.position = destination
        self.move_actions += 1
        return self.position

    def step(self):
        # one round, returns if it was found or not
        raise NotImplementedError

    def _nearest_path(self, candidates):
        # Search a complete BFS layer so all equally near targets can be chosen.
        queue = deque([self.position])
        parent = {self.position: None}
        targets = []
        while queue and not targets:
            for _ in range(len(queue)):
                cell = queue.popleft()
                if cell in candidates:
                    targets.append(cell)
                for neighbor in self.ship.get_open_neighbors(cell):
                    if neighbor not in parent:
                        parent[neighbor] = cell
                        queue.append(neighbor)
        if not targets:
            raise RuntimeError("No reachable possible LostBot location remains.")
        target = self.rng.choice(sorted(targets))
        path = []
        while target is not None:
            path.append(target)
            target = parent[target]
        return list(reversed(path))

    def run(self):
        # Search until a detector confirms discovery.
        while not self.step():
            pass
        return self.cost


class BaselineTypeABot(LocatorBot):
# pass rng seed so they work on the same ship
    def __init__(self, ship, start, lostBot, k, rng=None):
        super().__init__(ship, start, lostBot)
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer.")
        self.k = k
        self.rng = rng if rng is not None else random.Random()
        self.cells = np.array(sorted(self.open_cells), dtype=int)
        self._cell_index = {tuple(cell): i for i, cell in enumerate(self.cells)}
        # Boolean knowledge mask aligned with cells; includes the starting cell.
        self.possible_mask = np.ones(len(self.cells), dtype=bool)

    @property
    def possible_locations(self):
        """Candidate cells as a fresh set; assign a set to replace candidates."""
        return {tuple(cell) for cell in self.cells[self.possible_mask]}

    @possible_locations.setter
    def possible_locations(self, locations):
        indices = [self._cell_index[tuple(cell)] for cell in locations]
        self.possible_mask.fill(False)
        self.possible_mask[indices] = True

    def _in_range(self, cell):
        # Works for one coordinate or an entire (N, 2) coordinate array.
        return np.all(np.abs(np.asarray(cell) - self.position) <= self.k, axis=-1)

    def _detect(self):
        return (DetectorResult.PING if self._in_range(self._lost_bot)
                else DetectorResult.NO_PING)

    def update_knowledge(self, result):
        # Intersect the existing mask with the vectorized scan constraints.
        current = self._cell_index[self.position]
        if result is DetectorResult.FOUND:
            self.possible_mask.fill(False)
            self.possible_mask[current] = True
        elif result in (DetectorResult.PING, DetectorResult.NO_PING):
            in_range = self._in_range(self.cells)
            self.possible_mask &= (
                in_range if result is DetectorResult.PING else ~in_range
            )
            self.possible_mask[current] = False
        else:
            raise ValueError("Unknown detector result.")

    def _nearest_path(self, candidates=None):
        return super()._nearest_path(
            self.possible_locations if candidates is None else candidates
        )

    def step(self):
        if self.found:
            return True
        result = self.ping()
        self.update_knowledge(result)
        if self.found:
            return True
        for cell in self._nearest_path(self.possible_locations)[1:]:
            self.move(cell)
        return False



class BaselineTypeBBot(LocatorBot):
# bayesian bot based on maximum likelihood principle
# again pass rng in for the same ship
    def __init__(self, ship, start, lostBot, alpha, rng=None):
        super().__init__(ship, start, lostBot)
      
        self.alpha = float(alpha)
        self.rng = rng if rng is not None else random.Random()
        self.cells = np.array(sorted(self.open_cells), dtype=int)
        self._cell_index = {tuple(cell): i for i, cell in enumerate(self.cells)}
        self.log_belief = np.full(len(self.cells), -np.log(len(self.cells)))

    @property
    def probabilities(self):
        return np.exp(self.log_belief)

    def _log_ping_probabilities(self):
        distances = np.linalg.norm(self.cells - self.position, axis=1)
        # The current cell is handled by FOUND rather than the ping model.
        with np.errstate(over="ignore"):
            return -self.alpha * np.maximum(distances - 1.0, 0.0)

    def _detect(self):
        index = self._cell_index[self._lost_bot]
        probability = np.exp(self._log_ping_probabilities()[index])
        return (DetectorResult.PING if self.rng.random() < probability
                else DetectorResult.NO_PING)

    def update_knowledge(self, result):
        # Bayes: posterior(cell) is proportional to prior(cell) * likelihood.
        current = self._cell_index[self.position]
        if result is DetectorResult.FOUND:
            self.log_belief.fill(-np.inf)
            self.log_belief[current] = 0.0
            return
        if result not in (DetectorResult.PING, DetectorResult.NO_PING):
            raise ValueError("Unknown detector result.")
        log_ping = self._log_ping_probabilities()
        if result is DetectorResult.PING:
            likelihood = log_ping
        else:
            # log(1 - exp(log_ping)), accurate even when ping is nearly certain. (near 0 or 1)
            with np.errstate(divide="ignore"):
                likelihood = np.log(-np.expm1(log_ping))
        posterior = self.log_belief + likelihood
        posterior[current] = -np.inf  # A non-FOUND scan rules out this cell.
        maximum = np.max(posterior)

        # Log-sum-exp normalization
        posterior -= maximum
        posterior -= np.log(np.exp(posterior).sum())
        self.log_belief = posterior

    def _most_likely_path(self):
        maximum = np.max(self.log_belief)

        candidates = {
            tuple(cell) for cell in self.cells[self.log_belief == maximum]
        }
        return self._nearest_path(candidates)

    def step(self):
        if self.found:
            return True
        self.update_knowledge(self.ping())
        if self.found:
            return True
        for cell in self._most_likely_path()[1:]:
            self.move(cell)
        return False
