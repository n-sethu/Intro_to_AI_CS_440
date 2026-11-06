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
        destination = tuple(destination)
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
        else:
            in_range = self._in_range(self.cells)
            self.possible_mask &= (
                in_range if result is DetectorResult.PING else ~in_range
            )
            self.possible_mask[current] = False

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



# improved type A : scan for where info gained per unit of effort is highest
# as opposed to moving to nearest possible cell
# gain -> expected candidates eliminated
# cost -> distance to c + scan
# move_weight for free movement case later on
class ImprovedTypeABot(BaselineTypeABot):
    move_weight = 1.0
 
    def __init__(self, ship, start, lostBot, k, rng=None, move_weight=None):
        super().__init__(ship, start, lostBot, k, rng=rng)
        if move_weight is not None:
            self.move_weight = move_weight
 
    # gain -> how much does a scan at each cell shrink the candidate set
    # for each open cell : how many candidates are in its window? 
    def _window_counts(self):
        D = self.ship.D
        rows, cols = self.cells[:, 0], self.cells[:, 1]
        occupancy = np.zeros((D, D), dtype=np.int64)
        occupancy[rows, cols] = self.possible_mask
        integral = np.zeros((D + 1, D + 1), dtype=np.int64)
        integral[1:, 1:] = occupancy.cumsum(axis=0).cumsum(axis=1)
        r0 = np.clip(rows - self.k, 0, D)
        r1 = np.clip(rows + self.k + 1, 0, D)
        c0 = np.clip(cols - self.k, 0, D)
        c1 = np.clip(cols + self.k + 1, 0, D)
        return integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
 
    def _expected_gain(self):
        n = int(self.possible_mask.sum())
        n_in = self._window_counts()          # candidates inside the window
        n_out = n - n_in                      # candidates outside it
        own = self.possible_mask.astype(np.int64)  # standing on a candidate can FIND it
        # PING leaves (n_in - own) candidates, NO_PING leaves n_out, FOUND leaves 0.
        # Chances are n_in/n and n_out/n, so:  E[remaining] = numerator / n
        numerator = (n_in - own) ** 2 + n_out ** 2
        return n - numerator / n
 
    # cost -> walking distance from the current position to every cell
    def _bfs_all(self):
        dist = np.full(len(self.cells), np.inf)
        parent = {self.position: None}
        dist[self._cell_index[self.position]] = 0
        queue = deque([self.position])
        while queue:
            cell = queue.popleft()
            for neighbor in self.ship.get_open_neighbors(cell):
                if neighbor not in parent:
                    parent[neighbor] = cell
                    dist[self._cell_index[neighbor]] = dist[self._cell_index[cell]] + 1
                    queue.append(neighbor)
        return dist, parent
 
    def _choose_target(self):
        gain = self._expected_gain()
        dist, parent = self._bfs_all()
        score = gain / (self.move_weight * dist + 0.5)   # inf distance -> score 0
        best_score = score.max()
        best = np.flatnonzero(np.isclose(score, best_score, rtol=1e-12, atol=0.0))
        # tie-break: nearest first, then at random
        nearest = best[dist[best] == dist[best].min()]
        target = tuple(self.cells[self.rng.choice(nearest.tolist())])
        path = []
        while target is not None:
            path.append(target)
            target = parent[target]
        return list(reversed(path))
 
    def step(self):
        if self.found:
            return True
        for cell in self._choose_target()[1:]:
            self.move(cell)
        self.update_knowledge(self.ping())
        return self.found

 
# same strategy as improved type A
# difference is that move_weight is 0
class ImprovedTypeAFreeBot(ImprovedTypeABot):
    move_weight = 0.0
    
    
    
    
# free movement bots
# idea for type A - each detector action should split the search space in half (binary search type) so find the most optimal position in the area the bot could be which gets the most information and go there
# for type b - use a recursive bayesian update rule and go to the place that maximizes information (centroids not just highest probability)

class ImprovedTypeBBot(BaselineTypeBBot):
    """Batched one-step rollout with adaptive allocation of simulations.

    Defaults:
        initial_rollouts=8
        max_rollouts=64
        finalists=4
        max_candidates=32
        batch_size=16

    max_candidates=None considers every open cell.
    move_weight=0.0 handles the free-movement experiment.

    Finite simulations, shortlisting, and adaptive elimination do not guarantee
    improvement over baseline on every trial.
    """

    _geometry_cache = None

    def __init__(
        self, ship, start, lostBot, alpha, rng=None,
        initial_rollouts=4, max_rollouts=16, finalists=4,
        max_candidates=32, batch_size=16, planning_seed=0,
        move_weight=1.0
    ):
        from collections import OrderedDict
        from weakref import WeakKeyDictionary

        super().__init__(ship, start, lostBot, alpha, rng=rng)

        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")

        for name, value in (
            ("initial_rollouts", initial_rollouts),
            ("max_rollouts", max_rollouts),
            ("finalists", finalists),
            ("batch_size", batch_size),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")

        if initial_rollouts > max_rollouts:
            raise ValueError("initial_rollouts must be <= max_rollouts")

        if max_candidates is not None and (
            not isinstance(max_candidates, int) or max_candidates < 2
        ):
            raise ValueError(
                "max_candidates must be None or an integer >= 2"
            )

        if not np.isfinite(move_weight) or move_weight < 0:
            raise ValueError("move_weight must be finite and nonnegative")

        self.initial_rollouts = initial_rollouts
        self.max_rollouts = max_rollouts
        self.finalists = finalists
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self.move_weight = float(move_weight)

        # Planning never consumes the actual detector's random stream.
        self.planning_rng = np.random.default_rng(planning_seed)
        self._sensor_cache = OrderedDict()
        self._distance_cache = OrderedDict()
        # Bound per-bot caches by bytes. Small entry limits caused repeated
        # graph searches throughout the simulated continuations.
        n = len(self.cells)
        self._distance_cache_limit = max(1, (128 * 1024**2) // (n * 4))
        self._sensor_cache_limit = max(1, (128 * 1024**2) // (n * 16))

        self.last_action_values = {}
        self.last_action_samples = {}
        self.last_planned_cost = None

        # Reuse geometry for repeated trials/alpha values on the SAME ship object.
        # Weak references allow geometry to disappear when a ship is discarded.
        cls = ImprovedTypeBBot
        if cls._geometry_cache is None:
            cls._geometry_cache = WeakKeyDictionary()

        geometry = cls._geometry_cache.get(ship)
        if geometry is None:
            neighbors = [
                tuple(
                    self._cell_index[tuple(nb)]
                    for nb in ship.get_open_neighbors(tuple(cell))
                )
                for cell in self.cells
            ]
            geometry = {
                "neighbors": neighbors,
            }
            # Use compiled graph traversal when SciPy is available. Distances
            # stay exact; the Python BFS below remains a dependency-free fallback.
            try:
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import dijkstra
            except ImportError:
                pass
            else:
                sources = [i for i, adjacent in enumerate(neighbors) for _ in adjacent]
                destinations = [j for adjacent in neighbors for j in adjacent]
                geometry["graph"] = csr_matrix(
                    (np.ones(len(sources)), (sources, destinations)),
                    shape=(len(neighbors), len(neighbors)),
                )
                geometry["shortest_paths"] = dijkstra
            cls._geometry_cache[ship] = geometry

        self._geometry = geometry
        current = self._cell_index[self.position]

        if np.any(self._distances(current) < 0):
            raise ValueError("Ship must have connected open cells")

    @property
    def cost(self):
        return (
            self.move_weight * self.move_actions
            + 0.5 * self.detector_actions
        )

    def _distances(self, origin):
        """Cached BFS distances indexed by open-cell number."""
        cache = self._distance_cache

        if origin not in cache:
            if "graph" in self._geometry:
                exact = self._geometry["shortest_paths"](
                    self._geometry["graph"], indices=origin, unweighted=True,
                )
                distances = np.where(np.isfinite(exact), exact, -1).astype(np.int32)
            else:
                # Python lists avoid slow NumPy scalar access inside BFS.
                values = [-1] * len(self.cells)
                values[origin] = 0
                queue = [origin]
                for cell in queue:
                    next_distance = values[cell] + 1
                    for neighbor in self._geometry["neighbors"][cell]:
                        if values[neighbor] < 0:
                            values[neighbor] = next_distance
                            queue.append(neighbor)
                distances = np.asarray(values, dtype=np.int32)

            cache[origin] = distances

            if len(cache) > self._distance_cache_limit:
                cache.popitem(last=False)

        cache.move_to_end(origin)
        return cache[origin]

    def _sensor_logs(self, origin):
        """Log likelihoods; the sensing cell belongs to FOUND."""
        cache = self._sensor_cache

        if origin not in cache:
            distance = np.linalg.norm(
                self.cells - self.cells[origin], axis=1
            )

            with np.errstate(over="ignore", divide="ignore"):
                ping = -self.alpha * np.maximum(
                    distance - 1.0, 0.0
                )
                no_ping = np.log(-np.expm1(ping))

            ping[origin] = -np.inf
            no_ping[origin] = -np.inf

            cache[origin] = ping, no_ping
            if len(cache) > self._sensor_cache_limit:
                cache.popitem(last=False)

        cache.move_to_end(origin)
        return cache[origin]

    @staticmethod
    def _normalize_rows(weights):
        """Stable log-space normalization of a batch of beliefs."""
        maximum = np.max(weights, axis=1, keepdims=True)

        if not np.isfinite(maximum).all():
            raise ValueError(
                "Observation has zero probability under the belief"
            )

        shifted = weights - maximum
        return shifted - np.log(
            np.exp(shifted).sum(axis=1, keepdims=True)
        )

    def update_knowledge(self, result):
        current = self._cell_index[self.position]

        if result is DetectorResult.FOUND:
            self.log_belief.fill(-np.inf)
            self.log_belief[current] = 0.0
            return

        outcome = 0 if result is DetectorResult.PING else 1
        weights = (
            self.log_belief
            + self._sensor_logs(current)[outcome]
        )
        self.log_belief = self._normalize_rows(
            weights[None, :]
        )[0]

    @staticmethod
    def _top_indices(values, count):
        count = min(count, len(values))
        if count == len(values):
            return np.arange(count)

        return np.argpartition(
            values, len(values) - count
        )[-count:]

    def _candidates(self, current):
        n = len(self.cells)

        if self.max_candidates is None or self.max_candidates >= n:
            return np.arange(n)

        p = self.probabilities
        travel = self._distances(current)
        scores = np.empty(n)

        # Evaluate information gain in blocks, avoiding N x N temporaries.
        for first in range(0, n, 128):
            indices = np.arange(first, min(first + 128, n))
            delta = (
                self.cells[indices, None, :]
                - self.cells[None, :, :]
            )
            distance = np.sqrt(np.sum(delta * delta, axis=2))

            with np.errstate(
                over="ignore", divide="ignore", invalid="ignore"
            ):
                log_ping = -self.alpha * np.maximum(
                    distance - 1.0, 0.0
                )
                ping = np.exp(log_ping)
                no_ping = -np.expm1(log_ping)

                terms = np.where(
                    ping > 0, ping * log_ping, 0.0
                )
                terms += np.where(
                    no_ping > 0,
                    no_ping * np.log(no_ping),
                    0.0,
                )

            # Separate FOUND from PING at the sensing cell.
            q_ping = np.clip(
                ping @ p - p[indices], 0.0, 1.0
            )
            q_no = np.clip(no_ping @ p, 0.0, 1.0)
            outcomes = np.stack(
                (p[indices], q_ping, q_no), axis=1
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -np.where(
                    outcomes > 0,
                    outcomes * np.log(outcomes),
                    0.0,
                ).sum(axis=1)

            # Mutual information:
            # H(observation) - E_target[H(observation | target)].
            gain = np.maximum(entropy + terms @ p, 0.0)
            scores[indices] = gain / (
                self.move_weight * travel[indices] + 0.5
            )

        limit = self.max_candidates
        probable_count = max(1, (limit - 1) // 4)

        selected = {current}
        selected.update(
            self._top_indices(
                self.log_belief, probable_count
            ).tolist()
        )

        for action in np.argsort(-scores):
            selected.add(int(action))
            if len(selected) >= limit:
                break

        return np.array(sorted(selected), dtype=int)

    def _simulate_batch(self, current, action, targets):
        """Simulate one candidate action followed by baseline B.

        Each row has its own target, sampled from the current belief.
        This method never accesses self._lost_bot.
        """
        size = len(targets)
        n = len(self.cells)

        belief = np.broadcast_to(
            self.log_belief, (size, n)
        ).copy()
        positions = np.full(size, action, dtype=int)
        totals = np.full(
            size,
            self.move_weight * self._distances(current)[action],
        )
        rows = np.arange(size)

        # After the planned scan, baseline moves to the posterior maximum
        # before its next scan. There is no duplicate scan at this boundary.
        for _ in range(n + 1):
            totals[rows] += 0.5

            unfinished = positions != targets[rows]
            rows = rows[unfinished]
            positions = positions[unfinished]
            belief = belief[unfinished]

            if not len(rows):
                return totals

            unique, inverse = np.unique(
                positions, return_inverse=True
            )
            logs = [
                self._sensor_logs(int(pos)) for pos in unique
            ]

            log_ping = np.stack(
                [pair[0] for pair in logs]
            )[inverse]

            local_rows = np.arange(len(rows))
            ping_probability = np.exp(
                log_ping[local_rows, targets[rows]]
            )
            is_ping = (
                self.planning_rng.random(len(rows))
                < ping_probability
            )

            # Reuse this array for the observed log likelihoods.
            if not is_ping.all():
                log_no = np.stack(
                    [pair[1] for pair in logs]
                )
                log_ping[~is_ping] = log_no[
                    inverse[~is_ping]
                ]

            belief = self._normalize_rows(
                belief + log_ping
            )

            distance_rows = np.stack([
                self._distances(int(pos)) for pos in unique
            ])[inverse]

            # Exactly reproduce baseline B's tie rules:
            # highest probability, then nearest, then random.
            maximum = belief.max(axis=1, keepdims=True)
            tied = belief == maximum
            nearest = np.where(
                tied,
                distance_rows,
                np.iinfo(np.int32).max,
            ).min(axis=1)
            tied &= distance_rows == nearest[:, None]

            destination = tied.argmax(axis=1)

            for row in np.flatnonzero(
                tied.sum(axis=1) > 1
            ):
                destination[row] = self.planning_rng.choice(
                    np.flatnonzero(tied[row])
                )

            totals[rows] += (
                self.move_weight
                * distance_rows[local_rows, destination]
            )
            positions = destination

        raise RuntimeError(
            "Baseline continuation did not terminate"
        )

    def _choose_target(self):
        current = self._cell_index[self.position]
        travel = self._distances(current)
        support = np.flatnonzero(
            np.isfinite(self.log_belief)
        )

        # Exact solution when only one possible target remains.
        if len(support) == 1:
            target = int(support[0])
            value = float(
                self.move_weight * travel[target] + 0.5
            )
            self.last_planned_cost = value
            self.last_action_values = {
                tuple(self.cells[target]): value
            }
            self.last_action_samples = {
                tuple(self.cells[target]): 0
            }

        else:
            active = self._candidates(current).tolist()
            evaluated = active.copy()
            sums = np.zeros(len(self.cells))
            counts = np.zeros(len(self.cells), dtype=int)

            p = self.probabilities
            p /= p.sum()
            budget = self.initial_rollouts

            while True:
                additional = (
                    budget - int(counts[active[0]])
                )

                for first in range(
                    0, additional, self.batch_size
                ):
                    size = min(
                        self.batch_size, additional - first
                    )

                    # Shared target samples reduce differences caused
                    # solely by sampling easy targets for one action.
                    targets = self.planning_rng.choice(
                        len(self.cells), size=size, p=p
                    )

                    for action in active:
                        costs = self._simulate_batch(
                            current, action, targets
                        )
                        sums[action] += costs.sum()
                        counts[action] += size

                active.sort(
                    key=lambda a: (
                        sums[a] / counts[a],
                        travel[a],
                        a,
                    )
                )

                if budget >= self.max_rollouts:
                    break

                keep = min(
                    len(active),
                    max(
                        2,
                        self.finalists,
                        (len(active) + 1) // 2,
                    ),
                )
                survivors = active[:keep]

                # Preserve baseline's initial scan as a competitor.
                if (
                    current in active
                    and current not in survivors
                ):
                    survivors[-1] = current

                active = survivors
                budget = min(
                    2 * budget, self.max_rollouts
                )

            target = active[0]

            self.last_action_values = {
                tuple(self.cells[a]): float(
                    sums[a] / counts[a]
                )
                for a in evaluated
            }
            self.last_action_samples = {
                tuple(self.cells[a]): int(counts[a])
                for a in evaluated
            }
            self.last_planned_cost = float(
                sums[target] / counts[target]
            )

        # Walk downhill through distances to the destination.
        # Every returned step is a legal neighboring open cell.
        to_target = self._distances(target)
        path = [tuple(self.cells[current])]

        while current != target:
            current = next(
                nb
                for nb in self._geometry["neighbors"][current]
                if to_target[nb] == to_target[current] - 1
            )
            path.append(tuple(self.cells[current]))

        return path

    def step(self):
        if self.found:
            return True

        for cell in self._choose_target()[1:]:
            self.move(cell)

        self.update_knowledge(self.ping())
        return self.found