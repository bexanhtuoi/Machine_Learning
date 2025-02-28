# Input là mảng 2 chiều, vd: X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
class KMeans: # K-Means
  def __init__(self, k=3, espilon=1e-4, max_iter=300):
    self.k = k
    self.espilon = espilon
    self.max_iter = max_iter
    self.cost = 0

  def cost_function(self, X, y):
    cost = 0
    for i in range(self.k):
      cost += np.sum((X[y == i] - self.centroids[i])**2)
    return cost

  def fit(self, X):
    self.centroids = np.random.randn(self.k, X.shape[1])
    for _ in range(self.max_iter):
      old_centroids = self.centroids.copy()
      y = self.get_labels(X)
      self.update_centroids(X, y)

      if np.linalg.norm(self.centroids - old_centroids) < self.espilon:
        break
    
    self.cost, self.X, self.y = self.cost_function(X, y), X, y
    
  def get_labels(self, X):
    distance = np.zeros((X.shape[0], self.k)) #(n, k)
    for i, x in enumerate(X):
      for j, centroid in enumerate(self.centroids):
        distance[i, j] = np.sqrt(np.sum((x - centroid)**2))

    return np.argmin(distance, axis=1)

  def update_centroids(self, X, y):
    for i in range(self.k):
      self.centroids[i] = np.mean(X[y == i], axis=0)