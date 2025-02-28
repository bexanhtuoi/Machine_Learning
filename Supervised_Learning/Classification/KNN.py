# Input là mảng 2 chiều, vd: X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#                            y = np.array([[0], [1], [2]])
class KNN: # K-Nearest Neighbors 
  def __init__(self, k=3, distance_measure='Euclidean', use_for='Classification'):
    self.k = k
    self.distance_measure = distance_measure
    self.use_for = use_for # 'Classification' or 'Regression'
  
  def fit(self, X, y):
    self.X = X
    self.y = y

  def Euclide(self, X, Y):
    return np.sqrt(np.sum((X - Y)**2, axis=1))

  def Manhattan(self, X, Y):
    return np.sum(np.abs(X - Y), axis=1)

  def Squared_Euclide(self, X, Y):
    return self.Euclide(X, Y)**2

  def Cosine(self, X, Y):
    return 1 - (Y @ X / (np.sqrt(np.sum(X**2)) * np.sqrt(np.sum(Y**2))))

  def mode(self, y):
    mode_table = dict()
    for i in y:
      if i in mode_table:
        mode_table[i] += 1
      else:
        mode_table[i] = 1

    return max(mode_table, key=mode_table.get)


  def predict(self, X):
    y_pred = []

    for sample in X:
        list_distance = {'Euclidean': self.Euclide(sample, self.X),
                         'Manhattan': self.Manhattan(sample, self.X),
                         'Squared_Euclide': self.Squared_Euclide(sample, self.X),
                         'Cosine': self.Cosine(sample, self.X)} # Mặc dù sẽ chậm hơn if else vì phải tính cùng lúc hết các hàm nhưng nó sẽ gọn hơn

        distance = list_distance[self.distance_measure]
        k_indices = np.argsort(distance)[:self.k]
        prediction = self.mode(self.y[k_indices].reshape(-1)) if self.use_for == 'Classification' else np.mean(self.y[k_indices])
        y_pred.append(prediction)

    return np.array(y_pred)