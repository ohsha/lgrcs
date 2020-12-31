from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Node:

    def __init__(self,predicted_class, n_instances, n_samples_per_class):
        self.predicted_class = predicted_class
        self.n_instances = n_instances
        self.n_samples_per_class = n_samples_per_class
        self.threshold = 0
        self.feature_index = 0
        self.left = None
        self.right = None


class DecisionTree:

    def __init__(self, max_depth=3):
        self.max_depth = max_depth


    def _gini_prob(self, x, y):
        size_node = x.size

        p_j = np.array([x[y == c].size for c in range(self.n_classes)])
        P = (p_j / (size_node + 1e-5)) ** 2
        P_k = 1 - np.sum(P)

        return P_k


    def _gini(self,X, y, value):
        n_instances = X.shape[0]

        left = self._gini_prob(X[X <= value], y[X <= value])
        sum_of_left = len(X[X <= value])

        right = self._gini_prob(X[X > value], y[X > value])
        sum_of_right = len(X[X > value])

        gini_left = left * sum_of_left / (n_instances )
        gini_right = right * sum_of_right / (n_instances)

        return gini_left + gini_right, value


    def _best_split(self, X, y):
        best_idx, best_thr = None, None
        gini_list = np.array([self._gini(X[:,i], y, np.mean(X[:,i])) for i in range(self.n_features)])
        best_idx = np.argmin(gini_list[:,0])
        best_thr = gini_list[best_idx, 1]

        return best_idx, best_thr


    def _build_tree(self,X, y,depth=0):
        n_samples_per_class = [sum( y == c) for c in range(self.n_classes)]
        predicted_class = np.argmax(n_samples_per_class)

        node = Node(
            predicted_class=predicted_class,
            n_instances = X.shape[0],
            n_samples_per_class=n_samples_per_class
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                left_idx = X[:, idx] <= thr
                X_left, y_left = X[left_idx], y[left_idx]
                X_right, y_right = X[~left_idx], y[~left_idx]

                node.feature_index = idx
                node.threshold = thr

                if X_left.shape[0]:
                    node.left = self._build_tree(X_left, y_left, depth + 1)

                if X_right.shape[0]:
                    node.right = self._build_tree(X_right, y_right, depth + 1)

        return node


    def _predict(self, x, node):
        if x[node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right

        if node.left == None or node.right == None:
            return node.predicted_class
        else:
            return self._predict(x, node)


    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y )


    def predict(self, X_test):
        labels = [self._predict(x, self.tree) for x in X_test]
        return labels



if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(n_samples=500, n_features=7, shuffle=False, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = DecisionTree(max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc * 100.}%')

