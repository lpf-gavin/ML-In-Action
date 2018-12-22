import numpy as np
from .decision_tree_model import ClassificationTree

class RandomForest():
    """
    Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=100, min_samples_split=2, min_gain=0,
                 max_depth=float("inf"), max_features=None):

        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features

        self.trees = []
        # bulid forest
        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_samples_split=self.min_samples_split, min_impurity=self.min_gain,
                                      max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X, Y):
        # every tree use random data set(bootstrap) and random feature
        sub_sets = self.get_bootstrap_data(X, Y)
        n_features = X.shape[1]
        if self.max_features == None:
            print('max_features not set, use sqrt(feature number of data) by default)')
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            # get random features by Sampling with replacement
            sub_X, sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=True)
            sub_X = sub_X[:, idx]
            self.trees[i].fit(sub_X, sub_Y)
            self.trees[i].feature_indices = idx
            print("tree", i, "fit complete")

    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_X = X[:, idx]
            y_pre = self.trees[i].predict(sub_X)
            y_preds.append(y_pre)

        y_preds = np.array(y_preds).T

        y_pred = []
        for y_p in y_preds:
            # np.bincount()可以统计每个索引出现的次数
            # np.argmax()可以返回数组中最大值的索引
            # cheak np.bincount() and np.argmax() in numpy Docs
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return y_pred

    def get_bootstrap_data(self, X, Y):
        # get int(n_estimators) datas by bootstrap

        m = X.shape[0]
        Y = Y.reshape(m, 1)

        # combine X and Y for bootstrap easily
        X_Y = np.hstack((X, Y))
        np.random.shuffle(X_Y)

        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m, m, replace=True)
            bootstrap_X_Y = X_Y[idm, :]
            bootstrap_X = bootstrap_X_Y[:, :-1]
            bootstrap_Y = bootstrap_X_Y[:, -1:]
            data_sets.append([bootstrap_X, bootstrap_Y])
        return data_sets

