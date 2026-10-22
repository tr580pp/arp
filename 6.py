import numpy as np
import matplotlib.pyplot as plt

def differential_evolution(
    func, bounds, pop_size=50, F=0.8, CR=0.9, generations=200, strategy="rand/1", seed=None,
):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_val = fitness[best_idx]

    history = [best_val]
    diversity = [np.mean(np.std(pop, axis=0))]

    for gen in range(generations):
        for i in range(pop_size):
            idxs = np.arange(pop_size)
            idxs = idxs[idxs != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            a, b, c = pop[r1], pop[r2], pop[r3]
            if strategy == "rand/1":
                mutant = a + F * (b - c)
            elif strategy == "best/1":
                mutant = best + F * (b - c)
            elif strategy == "current-to-best/1":
                mutant = pop[i] + F * (best - pop[i]) + F * (b - c)
            else:
                raise ValueError("Unknown strategy: choose 'rand/1', 'best/1', or 'current-to-best/1'")
            
            mutant = np.clip(mutant, lower, upper)

            jrand = rng.integers(dim)
            trial = np.array([mutant[j] if (rng.random() < CR or j == jrand) else pop[i, j] for j in range(dim)])

            trial_val = func(trial)
            if trial_val < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_val
                if trial_val < best_val:
                    best_val = trial_val
                    best = trial.copy()
    
        history.append(best_val)
        diversity.append(np.mean(np.std(pop, axis=0)))
    
    return best, best_val, np.array(history), np.array(diversity)

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    A = 10.0
    x = np.asarray(x)
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

if __name__ == "__main__":
    dim = 10
    bounds = [(-5.12, 5.12)] * dim
    objective = rastrigin

    strategies = ["rand/1", "best/1", "current-to-best/1"]
    results = {}

    for s in strategies:
        best_vec, best_val, history, diversity = differential_evolution(
            func=objective, bounds=bounds, pop_size=60, F=0.7, CR=0.9,
            generations=300, strategy=s, seed=42,
        )
        results[s] = (history, diversity)
        print(f"{s:>20s} -> best value = {best_val:.6f}")
    
    plt.figure(figsize=(12, 5))
    for s, (history, _) in results.items():
        plt.plot(history, label=f"{s}")
    plt.title("Differential Evolution Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Objective Value")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 5))
    for s, (_, diversity) in results.items():
        plt.plot(diversity, label=f"{s}")
    plt.title("Population Diversity over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Std. Dev. of Population")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()