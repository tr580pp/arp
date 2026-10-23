import numpy as np
import matplotlib.pyplot as plt

def rastrigin(x):
    x = np.asarray(x)
    A = 10.0
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def firefly_algorithm(func,
                      dim=10,
                      n_fireflies=40,
                      max_iter=200,
                      lb=-5.12,
                      ub=5.12,
                      alpha=0.2,
                      alpha_decay=0.98,
                      beta0=1.0,
                      gamma=1.0,
                      seed=None):
    rng = np.random.default_rng(seed)

    X = rng.uniform(lb, ub, size=(n_fireflies, dim))
    fitness = np.array([func(x) for x in X])

    def brightness(f):
        return 1.0 / (1.0 + f)

    best_history = []
    best_idx = np.argmin(fitness)
    best_sol = X[best_idx].copy()
    best_fit = fitness[best_idx]
    best_history.append(best_fit)

    for t in range(max_iter):
        idx = np.argsort(fitness)
        X = X[idx]
        fitness = fitness[idx]

        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(X[i] - X[j])
                    beta = beta0 * np.exp(-gamma * (r**2))
                    step = beta * (X[j] - X[i])
                    random_vec = alpha * (rng.random(dim) - 0.5) * (ub - lb)
                    X[i] = X[i] + step + random_vec

                    X[i] = np.clip(X[i], lb, ub)

                    fitness[i] = func(X[i])

        cur_best_idx = np.argmin(fitness)
        if fitness[cur_best_idx] < best_fit:
            best_fit = fitness[cur_best_idx]
            best_sol = X[cur_best_idx].copy()

        best_history.append(best_fit)

        # optionally decay alpha
        alpha *= alpha_decay

    return best_sol, best_fit, best_history

if __name__ == "__main__":
    best_sol, best_fit, history = firefly_algorithm(
        func=rastrigin,
        dim=10,
        n_fireflies=40,
        max_iter=300,
        lb=-5.12,
        ub=5.12,
        alpha=0.5,
        alpha_decay=0.97,
        beta0=1.0,
        gamma=0.01,
        seed=42
    )

    print("Best fitness:", best_fit)
    print("Best solution (first 6 dims):", best_sol[:6])

    plt.figure(figsize=(8,5))
    plt.plot(history, linewidth=2)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (log scale)")
    plt.title("Firefly Algorithm Convergence (Rastrigin, dim=10)")
    plt.grid(True)
    plt.show()
