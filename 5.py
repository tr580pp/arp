import numpy as np
import matplotlib.pyplot as plt

class PSO:
    def __init__(
        self, func, dim, bounds, num_particles=30, max_iter=100,
        w=(0.9, 0.4), c1=1.49445, c2=1.49445, vmax=None, seed=None,
        verbose=False,
    ):
        self.func = func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)

        if isinstance(bounds, tuple) or (isinstance(bounds, list) and len(bounds) == 2 and not hasattr(bounds[0], "__len__")):
            low, high = bounds
            self.bounds = np.array([[low, high]] * dim, dtype=float)
        else:
            self.bounds = np.array(bounds, dtype=float)
            if self.bounds.shape != (dim, 2):
                raise ValueError("Bounds must match dimension")
        
        spans = self.bounds[:, 1] - self.bounds[:, 0]
        if vmax is None:
            self.vmax = spans.copy()
        else:
            self.vmax = np.array(vmax, dtype=float)
            if self.vmax.shape == ():
                self.vmax = np.repeat(self.vmax, dim)
        
        if isinstance(w, (list, tuple)) and len(w) == 2:
            self.w_max, self.w_min = float(w[0]), float(w[1])
        else:
            self.w_max = self.w_min = float(w)
        
        self.pos = self._init_positions()
        self.vel = self._init_velocities()

        self.fitness = np.array([self.func(p) for p in self.pos])

        self.pbest_pos = self.pos.copy()
        self.pbest_val = self.fitness.copy()

        gidx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[gidx].copy()
        self.gbest_val = float(self.pbest_val[gidx])
    
    def _init_positions(self):
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        r = np.random.rand(self.num_particles, self.dim)
        pos = lows + r * (highs - lows)
        return pos
    
    def _init_velocities(self):
        r = np.random.rand(self.num_particles, self.dim) * 2 - 1
        vel = r * self.vmax
        return vel
    
    def optimize(self):
        history = []
        for t in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * (t / max(1, self.max_iter - 1))

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)

            cognitive = self.c1 * r1 * (self.pbest_pos - self.pos)
            social = self.c2 * r2 * (self.gbest_pos - self.pos)

            self.vel = w * self.vel + cognitive + social
            self.vel = np.clip(self.vel, -self.vmax, self.vmax)

            self.pos = self.pos + self.vel
            lows = self.bounds[:, 0]
            highs = self.bounds[:, 1]
            self.pos = np.clip(self.pos, lows, highs)

            self.fitness = np.array([self.func(p) for p in self.pos])

            improved = self.fitness < self.pbest_val
            if np.any(improved):
                self.pbest_pos[improved] = self.pos[improved]
                self.pbest_val[improved] = self.fitness[improved]

            gidx = np.argmin(self.pbest_val)
            if self.pbest_val[gidx] < self.gbest_val:
                self.gbest_val = float(self.pbest_val[gidx])
                self.gbest_pos = self.pbest_pos[gidx].copy()
            
            history.append(self.gbest_val)

            if self.verbose and (t % max(1, self.max_iter // 10) == 0 or t == self.max_iter - 1):
                print(f"Iter {t+1}/{self.max_iter}, gBest = {self.gbest_val:.6e}")

        return self.gbest_pos, self.gbest_val, np.array(history)

def rastrigin(x):
    x = np.asarray(x)
    n = x.size
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))

if __name__ == "__main__":
    dim = 2
    bounds = [(-5.12, 5.12)] * dim
    pso = PSO(
        func=rastrigin,
        dim=dim,
        bounds=bounds,
        num_particles=40,
        max_iter=200,
        w=(0.9, 0.4),
        c1=1.49445,
        c2=1.49445,
        vmax=None,
        seed=42,
        verbose=True,
    )

    best_pos, best_val, history = pso.optimize()
    print("\nResume:")
    print("Best position:", best_pos)
    print("Best value (fitness):", best_val)

    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (gBest)")
    plt.title("PSO convergence (Rastrigin, 2D)")
    plt.grid(True)
    plt.show()
