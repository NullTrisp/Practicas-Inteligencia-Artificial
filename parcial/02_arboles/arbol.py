# create a decision tree
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

# load data
iris = load_iris()
X = iris.data
y = iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# create a decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# predict
y_pred = clf.predict(X_test)

# accuracy
print(accuracy_score(y_test, y_pred))

# visualize
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,
                                class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

graph.view()

print(iris.target_names[clf.predict([[5, 3, 1, 0.3]])])