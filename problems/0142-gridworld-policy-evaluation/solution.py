def gridworld_policy_evaluation(
    policy: dict, gamma: float, threshold: float
) -> list[list[float]]:
  """Evaluate state-value function for a policy on a 5x5 gridworld.

  Args:
      policy: dict mapping (row, col) to action probability dicts
      gamma: discount factor
      threshold: convergence threshold

  Returns:
      5x5 list of floats
  """
  grid_size = 5
  terminal_states = {(0, 0), (0, 4), (4, 0), (4, 4)}
  V = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]

  actions = {
      "up": (-1, 0),
      "down": (1, 0),
      "left": (0, -1),
      "right": (0, 1),
  }

  while True:
    delta = 0.0
    new_V = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]

    for r in range(grid_size):
      for c in range(grid_size):
        if (r, c) in terminal_states:
          continue

        v = 0.0
        for action, prob in policy[(r, c)].items():
          dr, dc = actions[action]
          nr, nc = r + dr, c + dc

          # Stay in place if hitting the grid boundary
          if nr < 0 or nr >= grid_size or nc < 0 or nc >= grid_size:
            nr, nc = r, c

          reward = -1.0
          v += prob * (reward + gamma * V[nr][nc])

        new_V[r][c] = v
        delta = max(delta, abs(v - V[r][c]))

    V = new_V
    if delta < threshold:
      break

  return V
