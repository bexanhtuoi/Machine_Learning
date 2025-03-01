def PCA(X, dimension: int):
  n, d = X.shape
  #Chuẩn hóa dữ liệu
  _mean = []
  _std = []
  for i in X.T:
    _mean.append(np.mean(i))
    _std.append(np.std(i))
  X = (X - _mean) / _std

  #Tính ma trận hiệp phương sai
  _mean = [np.mean(i) for i in X.T]

  X = X - _mean
  cov = 1/n * X.T @ X

  #Tính trị riêng và vector riêng
  eigenvalues, eigenvectors = np.linalg.eig(cov)

  return X @ eigenvectors[:, :dimension]
