from example_package.example import EventCorrelationClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

neigh = EventCorrelationClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1.1]]))