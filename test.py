import numpy as np

rng = np.random.default_rng(1234)
a_mat = rng.uniform(0, 1, (2, 2))

context = rng.uniform(0, 1, (2, 1000))
temp = context.T @ a_mat.T @ a_mat @ context
diag = np.diag(temp) if isinstance(temp, np.ndarray) else temp
fomp = 0.5 * np.cos(diag) + 0.5
print("chomp")
# return np.clip(diag, 0, 1)