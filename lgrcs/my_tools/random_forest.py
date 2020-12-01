import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class DecisionTree():

    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def _gini_prob(self, x, y):
        #gini = 1 - np.sum((np.sum(x == c) / x.size)**2 for c in range(2))
        size = x.size
        if size == 0:
            return 0
        left = x[y == 0].size
        right = x[y == 1].size

        gini = 1 - (left / size)**2 - (right / size)**2
        return gini

    def _gini(self,X, y, value):
        n_instances = X.shape[0]
        if n_instances == 0:
            return 0

        left = self._gini_prob(X[X <= value], y[X <= value])
        sum_of_left = len(X[X <= value])

        right = self._gini_prob(X[X > value], y[X > value])
        sum_of_right = len(X[X > value])

        gini_left = left * sum_of_left / n_instances
        gini_right = right * sum_of_right / n_instances

        return gini_left + gini_right, value

    def _best_split(self, X, y):
        best_idx, best_thr = None, None
        X_t = X.values
        gini_list = np.array([self._gini(X.iloc[:,i], y, np.mean(X.iloc[:,i])) for i in range(self.n_features)])
        best_idx = np.argmin(gini_list[:,0])
        best_thr = gini_list[best_idx, 1]

        return best_idx, best_thr

    def _build_tree(self,X, y,depth=0):

        n_sampels_per_class = [sum( y == c) for c in range(self.n_classes)]
        predicted_class = np.argmax(n_sampels_per_class)

        node = Node(

            predicted_class=predicted_class,
            n_instances = X.shape[0],
            n_sampels_per_class=n_sampels_per_class
        )
        if depth < self.max_depth:

            idx, thr = self._best_split(X, y)
            if idx is not None:
                left_idx = X.iloc[:, idx] <= thr

                X_left, y_left = X.loc[left_idx], y.loc[left_idx]
                X_right, y_right = X.loc[~left_idx], y.loc[~left_idx]
                node.feature_index = list(X)[idx]
                node.threshold = thr
                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, test_instance):

        node = self.tree
        while node.left:
            if test_instance[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y )

    def predict(self, X_test):
        label_list = [self._predict(instance) for instance in X_test]
        return label_list


class Node():
    def __init__(self,predicted_class, n_instances, n_sampels_per_class):
        self.predicted_class = predicted_class
        self.n_instances = n_instances
        self.n_sampels_per_class = n_sampels_per_class
        self.threshold = 0
        self.feature_index = 0
        self.left = None
        self.right = None


class RandomForest():

    def __init__(self, max_tree=4, tree_depth=2):

        self.max_tree = max_tree
        self.tree_depth = tree_depth


    def _build_forest(self, X, y):

        tree_list = np.array([])

        for k in range(self.max_tree):

            n_instances_samples = self.n_instances // 1
            n_features_samples = int(np.sqrt(self.n_features))

            random_instances = np.random.choice(self.n_instances, n_instances_samples, replace=True)
            random_features = np.random.choice(self.n_features, n_features_samples, replace=True)

            X_rand= X.iloc[random_instances,:]
            X_rand = X_rand.iloc[:, random_features]
            y_rand = y.iloc[random_instances]
            #X_rand = X_rand[:, [random_features]]
            dt = DecisionTree(max_depth=self.tree_depth)
            dt.fit(X_rand, y_rand)
            tree_list = np.append(tree_list, dt.tree)

        return tree_list

    def _predict(self, instance):
        predicted_list = []

        for i in range(self.forest.shape[0]):
            node = self.forest[i]
            while node.left:
                if instance[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predicted_list.append(node.predicted_class)

        return self._get_majority_decision(predicted_list)

    def _get_majority_decision(self, votes):
        votes = np.array(votes)
        sum_votes = [sum(votes == c) for c in range(self.n_classes)]

        return np.argmax(sum_votes)

    def predict(self, X_test):
        return [self._predict(X_test.iloc[instance,:]) for instance in range(X_test.shape[0])]

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.n_instances = X.shape[0]
        self.forest = self._build_forest(X, y)


if __name__ == "__main__":
    data = pd.read_csv('datasets\wdbc.data', header=None)
    data = data.replace(to_replace=['M'], value=1, inplace=False)
    data = data.replace(to_replace=['B'], value=0, inplace=False)
    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    rf = RandomForest(max_tree=50, tree_depth=2)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = (y_pred == y_test).astype(np.int).sum() / y_test.size
