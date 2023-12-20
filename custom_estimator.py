import numpy as np
import pandas as pd
import pydot
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_size=10,class_weight=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.class_weight = class_weight
        self.tree_ = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)  # Add this line to set classes_
        node_indices = np.arange(X.shape[0])
        self.tree_ = self._build_tree_recursive(X, y, node_indices, depth=0)
        if self.class_weight in ['balanced', None]:
            # Compute the class weights if needed
            class_weights = compute_class_weight(self.class_weight, classes=np.unique(y), y=y)
            self.class_weight_ = dict(enumerate(class_weights))
        else:
            self.class_weight_ = self.class_weight

        return self

    def predict(self, X):
        X = np.array(X)
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            predictions[i] = self._predict_sample(X[i, :], self.tree_)
        return predictions

    def _build_tree_recursive(self, X, y, node_indices, depth):
        if len(node_indices) <= self.min_size or depth >= self.max_depth:
            return self._find_leaf_node(y[node_indices])
        best_feature, best_info_gain = self._find_best_split(X, y, node_indices)
        if best_info_gain <= 0:
            return self._find_leaf_node(y[node_indices])
        left_indices, right_indices = self._split_dataset(X, node_indices, best_feature)
        if not left_indices.size or not right_indices.size:
            return self._find_leaf_node(y[node_indices])
        left_branch = self._build_tree_recursive(X, y, left_indices, depth + 1)
        right_branch = self._build_tree_recursive(X, y, right_indices, depth + 1)
        return {"best_feature": best_feature, "left_branch": left_branch, "right_branch": right_branch}

    def _predict_sample(self, x, tree):
        if isinstance(tree, int) or isinstance(tree, np.int64):
            return tree
        feature = tree["best_feature"]
        if x[feature] <= 0.5:
            return self._predict_sample(x, tree["left_branch"])
        else:
            return self._predict_sample(x, tree["right_branch"])

    def _compute_entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _split_dataset(self, X, node_indices, feature):
        left_indices = [idx for idx in node_indices if X[idx, feature] <= 0.5]
        right_indices = [idx for idx in node_indices if X[idx, feature] > 0.5]
        return np.array(left_indices), np.array(right_indices)

    def _compute_information_gain(self, X, y, node_indices, feature):
        entropy_node = self._compute_entropy(y[node_indices])
        left_indices, right_indices = self._split_dataset(X, node_indices, feature)
        if left_indices.size == 0 or right_indices.size == 0:
            return 0
        entropy_left = self._compute_entropy(y[left_indices])
        entropy_right = self._compute_entropy(y[right_indices])
        weighted_entropy = (len(left_indices) / len(node_indices) * entropy_left +
                            len(right_indices) / len(node_indices) * entropy_right)
        return entropy_node - weighted_entropy

    def _find_best_split(self, X, y, node_indices):
        best_info_gain = 0
        best_feature = 0
        for feature in range(X.shape[1]):
            info_gain = self._compute_information_gain(X, y, node_indices, feature)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
        return best_feature, best_info_gain

    def _find_leaf_node(self, y):
        return np.bincount(y).argmax()

from sklearn.preprocessing import LabelEncoder
# Usage example
def main():
    data = pd.read_csv('./spotify_songs.csv')
    data = data[:29000]

    # Convert 'track_popularity' into three categories: Low, Medium, High
    popularity_bins = [0, 33, 66, 100]
    popularity_labels = [0,1,2]
    data['track_popularity'] = pd.cut(data['track_popularity'], bins=popularity_bins, labels=popularity_labels, include_lowest=True, right=True)
    

    X = data.drop(columns=['track_id', 'track_name', 'track_album_id', 'track_artist', 'playlist_name', 
                           'playlist_genre', 'playlist_subgenre', 'key', 'mode', 'track_popularity', 
                           'track_album_name', 'track_album_release_date', 'playlist_id',
                           'duration_ms'])
    # print(X.columns)
    # exit()
    y = data['track_popularity']

    # Encode categorical features and the target variable
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.2, random_state=42)

    # # Check the distribution of classes in the training set
    # train_class_distribution = pd.Series(y_train).value_counts()
    # print(train_class_distribution)

    # # Check the distribution of classes in the testing set
    test_class_distribution = pd.Series(y_test).value_counts()
    print(test_class_distribution)




    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    class_counts = y_train.value_counts()
    min_class_size = class_counts.min()

    # Ensure that k_neighbors is at least 1 and less than the smallest class size
    min_number_of_neighbors = max(1, min(5, min_class_size - 1))

    sm = SMOTE(random_state=42, k_neighbors=min_number_of_neighbors)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # param_grid = {
    #     'max_depth': [3, 5, 7, 10, 15, 20],
    #     'min_size': [50, 100, 150, 200, 250, 300]
    # }

    # # Initialize GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=CustomDecisionTreeClassifier(),
    #     param_grid=param_grid,
    #     cv=5,  # Number of cross-validation folds
    #     scoring='f1_weighted',  # You can change this to another scoring method if needed
    #     n_jobs=-1  # Use all available cores
    # )

    # # Fit GridSearchCV
    # grid_search.fit(X_train_sm, y_train_sm)

    # # Extract best parameters and score
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_

    # print("Best Parameters:", best_params)
    # print("Best F1 Score:", best_score * 100)


    clf = CustomDecisionTreeClassifier(max_depth=5, min_size=10, class_weight='weighted')    
    clf.fit(X_train_sm, y_train_sm)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy * 100)
    print("F1 Score:", f1 * 100)

    # scores = cross_val_score(clf, X, y, cv=5)
    # print("Cross-validated scores:", scores)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=50)
    tree.fit(X_train_sm, y_train_sm)
    y_pred = tree.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy * 100)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1:", f1*100)


if __name__ == "__main__":
    main()
