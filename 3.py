import numpy as np
import math
import random
import matplotlib.pyplot as plt

def euclidean_distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def tour_length(route, dist_matrix):
    n = len(route)
    return sum(dist_matrix[route[i], route[(i + 1) % n]] for i in range(n))

def two_opt_swap(route):
    a, b = sorted(random.sample(range(len(route)), 2))
    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
    return new_route

def simulated_annealing(coords, T0=100, Tmin=1e-3, alpha=0.995, max_iter=20000):
    n = len(coords)
    dist_matrix = euclidean_distance_matrix(coords)

    route = list(range(n))
    random.shuffle(route)
    best_route = route[:]
    best_cost = tour_length(route, dist_matrix)
    current_cost = best_cost

    T = T0
    history = [best_cost]

    for it in range(max_iter):
        candidate = two_opt_swap(route)
        candidate_cost = tour_length(candidate, dist_matrix)
        delta = candidate_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            route = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = route[:]
        
        history.append(best_cost)

        T *= alpha
        if T < Tmin:
            break
    
    return best_route, best_cost, history

if __name__ == "__main__":
    np.random.seed(42)
    coords = np.random.rand(30, 2) * 100

    best_route, best_cost, history = simulated_annealing(coords)

    print(f"Best tour length: {best_cost:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.title("Simulated Annealing Progress")
    plt.show()

    plt.figure(figsize=(6, 6))
    ordered_coords = coords[best_route + [best_route[0]]]
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-', markersize=8)
    plt.title(f"Final Tour (Length = {best_cost:.2f})")
    plt.show()
    
