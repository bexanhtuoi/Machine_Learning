# Input là mảng 2 chiều, vd: X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
class GMM: # Gaussian Mixture Models
  def __init__(self, k=2, max_iter=100):
    self.k = k
    self.max_iter = max_iter

  def init_parameters(self, X):
    N, M = X.shape
    self.mu = np.random.randn(self.k, M)
    self.sigma = np.random.randn(self.k, M)
    self.pi = np.random.dirichlet(np.ones(self.k))
    
  def Gaussian(self, x, mu, sigma):
    return np.prod(1 / (np.sqrt(2 * np.pi * (sigma)**2)) * np.exp(-(x - mu)**2 / (2 * sigma**2)))

  def e_step(self, X):
    N, M = X.shape
    self.r = np.zeros((N, self.k))
    for i in range(N):
      for j in range(self.k):
        self.r[i, j] = self.pi[j] * self.Gaussian(X[i], self.mu[j], self.sigma[j])
      self.r[i] /= np.sum(self.r[i])

  def m_step(self, X):
    N, M = X.shape
    self.pi = np.sum(self.r, axis=0) / N
    for j in range(self.k):
      self.mu[j] = np.sum(self.r[:, j].reshape(-1, 1) * X, axis=0) / np.sum(self.r[:, j])
      self.sigma[j] = np.sqrt(np.sum(self.r[:, j].reshape(-1, 1) * (X - self.mu[j])**2, axis=0) / np.sum(self.r[:, j]))


  def fit(self, X):
    N, M = X.shape
    self.init_parameters(X)
    for i in range(self.max_iter):
      self.e_step(X)
      old_mu = self.mu.copy()
      self.m_step(X)
      
      if np.sqrt(np.sum((self.mu - old_mu)**2)) < 1e-4:
        break
    return np.argmax(self.r, axis=1)
