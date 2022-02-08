from sklearn import tree,ensemble,neighbors

KNN_NEIGHBOUR = 6

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "rf": ensemble.RandomForestClassifier(),
    "knn": neighbors.KNeighborsClassifier(n_neighbors=KNN_NEIGHBOUR)
}
