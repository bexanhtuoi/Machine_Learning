def PCA(X, dimension: int):
  n, d = X.shape
  _mean = []
  _std = []
  for i in X.T:
    _mean.append(np.mean(i))
    _std.append(np.std(i))
  X = (X - _mean) / _std

  X = X - _mean
  cov = 1/n * X.T @ X
  eigenvalues, eigenvectors = np.linalg.eig(cov)

  return X @ eigenvectors[:, :dimension]
