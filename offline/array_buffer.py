import numpy as np
import torch

def validate_arrays(s, a, r, done, s2, s_dim, a_dim):
    arrays = [np.asarray(x) for x in (s, a, r, done, s2)]
    s, a, r, done, s2 = arrays
    n = len(s)
    expected = [(n, s_dim), (n, a_dim), (n, 1), (n, 1), (n, s_dim)]
    for name, x, shape in zip(("s", "a", "r", "terminated", "s2"), arrays, expected):
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(f"{name}: expected finite shape {shape}, got {x.shape}")
    if np.any(np.abs(a)>1.) or not np.all((done == 0) | (done == 1)):
        raise ValueError("Actions must be normalized to [-1,1]; terminated must be binary")
    return [x.astype(np.float32, copy=False) for x in arrays]

class ArrayReplayBuffer:
    def __init__(self, s_dim, a_dim, capacity, device="cpu"):
        self.s_dim, self.a_dim = int(s_dim), int(a_dim)
        self.capacity, self.device = int(capacity), device
        self._n = self._ptr = 0
        self._allocate(0)

    def _allocate(self, n):
        self.s = np.zeros((n, self.s_dim), dtype=np.float32)
        self.a = np.zeros((n, self.a_dim), dtype=np.float32)
        self.r = np.zeros((n, 1), dtype=np.float32)
        self.done = np.zeros((n, 1), dtype=np.float32)
        self.s2 = np.zeros((n, self.s_dim), dtype=np.float32)
        self.w = None

    def __len__(self):
        return self._n

    def load_from_arrays(self, s, a, r, done, s2, w=None):
        values = validate_arrays(s, a, r, done, s2, self.s_dim, self.a_dim)
        n = len(s)
        if n < 1 or n > self.capacity:
            raise ValueError(f"Dataset size {n} outside [1,{self.capacity}]")
        self._allocate(n)
        self.s[:], self.a[:], self.r[:], self.done[:], self.s2[:] = values
        self._n, self._ptr = n, n % self.capacity
        if w is not None:
            self.set_weights(w)

    def set_weights(self, w):
        w = np.asarray(w, dtype=np.float32)
        if w.shape != (self._n, 1) or not np.all(np.isfinite(w)) or np.any(w<0):
            raise ValueError(f"Expected nonnegative finite source weights ({self._n},1)")
        self.w = w.copy()

    def sample_indices(self, batch_size):
        if self._n == 0 or batch_size < 1:
            raise RuntimeError("Cannot sample empty buffer or nonpositive batch")
        return np.random.randint(0, self._n, size=int(batch_size))

    def get_batch_by_indices(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        values = tuple(torch.from_numpy(x[idx]).to(self.device) for x in (self.s, self.a, self.r, self.done, self.s2))
        return values+(None if self.w is None else torch.from_numpy(self.w[idx]).to(self.device),)

    def sample(self, batch_size):
        return self.get_batch_by_indices(self.sample_indices(batch_size))

    def all(self):
        return self.s[:self._n], self.a[:self._n], self.r[:self._n], self.done[:self._n], self.s2[:self._n]
