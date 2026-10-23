import numpy as np
import matplotlib.pyplot as plt

def sphere(x):
    return np.sum(x ** 2)

def BBO(func, dim=5, pop_size=20, generations=100, bounds=(-5, 5),
        elite_count=2, mutation_rate=0.05):
    
    lower, upper = bounds
    pop = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])

    I, E = 1.0, 1.0
    best_curve = []

    for gen in range(generations):
        idx = np.argsort(fitness)
        pop, fitness = pop[idx], fitness[idx]

        fit_range = fitness.max() - fitness.min()
        if fit_range == 0:
            norm_fit = np.ones_like(fitness)
        else:
            norm_fit = (fitness.max() - fitness) / (fit_range + 1e-9)

        lambda_ = I * (1 - norm_fit)
        mu = E * norm_fit

        new_pop = np.copy(pop)

        for i in range(pop_size):
            for j in range(dim):
                if np.random.rand() < lambda_[i]:
                    if mu.sum() == 0:
                        donor = np.random.randint(0, pop_size)
                    else:
                        probs = mu / mu.sum()
                        donor = np.random.choice(pop_size, p=probs)
                    new_pop[i, j] = pop[donor, j]

        for i in range(elite_count, pop_size):
            for j in range(dim):
                if np.random.rand() < mutation_rate:
                    new_pop[i, j] = np.random.uniform(lower, upper)

        new_fitness = np.array([func(ind) for ind in new_pop])

        elites, elite_fit = pop[:elite_count].copy(), fitness[:elite_count].copy()
        worst = np.argsort(new_fitness)[-elite_count:]
        new_pop[worst], new_fitness[worst] = elites, elite_fit
        pop, fitness = new_pop, new_fitness

        best_curve.append(fitness.min())

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], best_curve

np.random.seed(42)
best_solution, best_value, best_curve = BBO(sphere, dim=5, generations=200)

plt.figure(figsize=(8, 5))
plt.plot(best_curve, linewidth=2)
plt.title("BBO Convergence Curve (Sphere Function)")
plt.xlabel("Generations")
plt.ylabel("Best Fitness Value")
plt.grid(True)
plt.show()

print("Best Solution:", best_solution)
print("Best Fitness:", best_value)
