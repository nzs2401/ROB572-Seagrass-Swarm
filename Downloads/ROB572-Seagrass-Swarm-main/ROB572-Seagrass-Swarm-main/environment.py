import numpy as np

COVERAGE_NONE       = 0
COVERAGE_PRESENT    = 1
COVERAGE_PATCHY     = 2
COVERAGE_CONTINUOUS = 3

DEPTH_MIN = 1.0
DEPTH_MAX = 3.0

def planting_likelihood(depth: float, coverage: int) -> float:
    if not (DEPTH_MIN <= depth <= DEPTH_MAX):
        return 0.0
    mapping = {
        COVERAGE_CONTINUOUS: 0.00,
        COVERAGE_PRESENT:    0.33,
        COVERAGE_PATCHY:     0.66,
        COVERAGE_NONE:       1.00,
    }
    return mapping.get(coverage, 0.0)

def likelihood_to_class(val: float) -> int:
    if val <= 0.0:   return 0
    elif val <= 0.33: return 1
    elif val <= 0.66: return 2
    else:             return 3

class Environment:
    def __init__(self, depth_grid: np.ndarray, coverage_grid: np.ndarray):
        assert depth_grid.shape == coverage_grid.shape
        self.depth_grid    = depth_grid.astype(float)
        self.coverage_grid = coverage_grid.astype(int)
        self.rows, self.cols = depth_grid.shape
        self.likelihood_grid = np.zeros((self.rows, self.cols), dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                self.likelihood_grid[r, c] = planting_likelihood(
                    self.depth_grid[r, c], self.coverage_grid[r, c]
                )

    def sense_depth(self, r, c):
        r, c = self._clip(r, c)
        return self.depth_grid[r, c]

    def sense_coverage(self, r, c):
        r, c = self._clip(r, c)
        return self.coverage_grid[r, c]

    def sense_likelihood(self, r, c):
        r, c = self._clip(r, c)
        return self.likelihood_grid[r, c]

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _clip(self, r, c):
        return int(np.clip(r, 0, self.rows-1)), int(np.clip(c, 0, self.cols-1))