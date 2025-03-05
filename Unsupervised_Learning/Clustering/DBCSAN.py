# Input là mảng 2 chiều, vd: X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
class DBSCAN: # DBSCAN
  def __init__(self, eps, min_samples):
    self.eps = eps
    self.min_samples = min_samples

  def Euclidean(self, X):
    N = X.shape[0]
    dist = np.zeros((N, N))

    for i in range(N):
      for j in range(i, N):
        dist[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
        dist[j, i] = dist[i, j]

    return dist

  def find_neighbors(self, dist):
    N = dist.shape[0]
    neighbors = []

    for point in dist:
      neighbor = [i for i in range(len(point)) if point[i] <= self.eps]
      neighbors.append(neighbor)

    return neighbors

  def find_core(self, neighbors):
    return [len(i) >= self.min_samples for i in neighbors]


  def process_cluster(self, arr, neighbors, cores, cluster):
    for i in arr:
      if self.y[i] == 0:
        self.y[i] = cluster
        if cores[i]:
          self.process_cluster(neighbors[i], neighbors, cores, cluster)


  def labeling(self, X, neighbors, cores):
    N = X.shape[0]
    self.y = np.zeros(N)
    cluster = 1

    for i in range(N):
      if self.y[i] != 0 or not cores[i]:
        continue

      self.process_cluster(neighbors[i], neighbors, cores, cluster)
      
      cluster += 1


  def fit(self, X):
     dist = self.Euclidean(X)
     neighbors = self.find_neighbors(dist)
     cores = self.find_core(neighbors)
     self.labeling(X, neighbors, cores)

     return self.y # Những điểm bằng 0 là ngoại lai