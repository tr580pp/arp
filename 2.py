import random

def fitness(state):
  n = len(state)
  total_pairs = n * (n - 1) // 2
  attacking = 0
  for i in range(n):
    for j in range(i + 1, n):
      if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
        attacking += 1
  return total_pairs - attacking

def random_state(n):
  return [random.randint(0, n - 1) for _ in range(n)]

def get_neighbors(state):
  neighbors = []
  n = len(state)
  for col in range(n):
    for row in range(n):
      if state[col] != row:
        neighbor = list(state)
        neighbor[col] = row
        neighbors.append(neighbor)
  return neighbors

def hill_climbing(n):
  current = random_state(n)
  current_fitness = fitness(current)
  while True:
    neighbors = get_neighbors(current)
    if not neighbors:
      break
    neighbor = max(neighbors, key=lambda x: fitness(x))
    neighbor_fitness = fitness(neighbor)
    if neighbor_fitnes <= current_fitness:
      break
    current, current_fitness = neighbor, neighbor_fitness
  return current, current_fitness

def random_restart_hill_climbing(n, max_restarts=1000):
  best_solution = None
  best_fitness = -1
  goal_fitness = (n * (n - 1)) // 2
  for restart in range(max_restarts):
    solution, score = hill_climbing(n)
    if score > best_fitness:
      best_solution, best_fitness = solution, score
    if best_fitness == goal_fitness:
      print(f"Solution found after {restart+1} restarts")
      return best_solution, best_fitness
  print("Max restarts reached, best solution found so far")
  return best_solution, best_fitness

def print_board(state):
  n = len(state)
  for row in range(n):
    line = ""
    for col in range(n):
      if state[col] == row:
        line += " Q "
      else:
        line += " . "
    print(line)
  print("\n")

if __name__ == "__main__":
  N = 8
  solution, score = random_restart_hill_climbing(N, max_restarts=500)
  print("Final Solution (row positions of queens):", solution)
  print("Fitness (non-attacking pairs):", score)
  if score == (N * (N - 1)) // 2:
    print("Valid Solution Found")
  else:
    print("Local Maxima but Best Found")
  print("\nVisual Board:")
  print_board(solution)
