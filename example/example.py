from scikit_event_correlation.classifiers import EventCorrelationClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

model = EventCorrelationClassifier()
model.fit(X, y)
print(model.predict([[1.1]]))