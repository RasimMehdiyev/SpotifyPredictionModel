import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# decision_trees.py

def compute_entropy(y):
    """
    Compute entropy of a vector y.
    """
    # Compute counts of each label
    counts = np.bincount(y)
    # Divide by the total observations to get probabilities
    probabilities = counts / len(y)
    # Initialize entropy
    entropy = 0
    # Loop through probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            entropy += prob * np.log2(prob)
    return -entropy

def split_dataset(X, y, node_indices, feature):
    # Initialize lists to store indices of left and right nodes
    left_indices = []
    right_indices = []

    # Iterate over node_indices to determine the split
    for idx in node_indices:
        if X.iloc[idx, feature] <= 0.5:
            left_indices.append(idx)
        else:
            right_indices.append(idx)

    return np.array(left_indices), np.array(right_indices)




def compute_information_gain(X, y, node_indices, feature):
    entropy_node = compute_entropy(y[node_indices])

    left_indices, right_indices = split_dataset(X, y, node_indices, feature)

    # Handle cases where there are no samples in one of the splits
    if left_indices.size == 0 or right_indices.size == 0:
        return 0

    entropy_left = compute_entropy(y[left_indices])
    entropy_right = compute_entropy(y[right_indices])

    weighted_entropy = (len(left_indices) / len(node_indices) * entropy_left +
                        len(right_indices) / len(node_indices) * entropy_right)
    
    return entropy_node - weighted_entropy



def find_best_split(X, y, node_indices):
    """
    Find the best split for a node given a dataset X, target y, and
    indices of data points in the node.
    """
    # Initialize variables to store best information gain and best feature
    best_info_gain = 0
    best_feature = 0
    # Iterate over features
    for feature in range(X.shape[1]):
        # Compute information gain
        info_gain = compute_information_gain(X, y, node_indices, feature)
        # Update best_info_gain and best_feature if info_gain is larger than best_info_gain
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature, best_info_gain

def find_leaf_node(y):
    """
    Find the leaf node given target y.
    """
    # Choose the most common label
    return np.bincount(y).argmax()

def build_tree_recursive(X, y, node_indices, max_depth, min_size, depth):
    if len(node_indices) <= min_size or depth >= max_depth:
        return find_leaf_node(y[node_indices])

    best_feature, best_info_gain = find_best_split(X, y, node_indices)
    if best_info_gain <= 0:
        return find_leaf_node(y[node_indices])

    left_indices, right_indices = split_dataset(X, y, node_indices, best_feature)
    
    # Handle cases where the split doesn't divide the node
    if len(left_indices) == 0 or len(right_indices) == 0:
        return find_leaf_node(y[node_indices])

    left_branch = build_tree_recursive(X, y, left_indices, max_depth, min_size, depth + 1)
    right_branch = build_tree_recursive(X, y, right_indices, max_depth, min_size, depth + 1)

    return {"best_feature": best_feature, "left_branch": left_branch, "right_branch": right_branch}




def build_tree(X, y, max_depth, min_size):
    """
    Build a decision tree given a dataset X, target y, maximum depth,
    and minimum size of leaf nodes.
    """
    # Initialize indices of data points in the root node
    node_indices = np.arange(X.shape[0])
    # Build the tree recursively
    tree = build_tree_recursive(X, y, node_indices, max_depth, min_size, depth=0)
    return tree

def predict_sample(x, tree):
    if isinstance(tree, int) or isinstance(tree, np.int64):
        return tree

    feature = tree["best_feature"]
    # Use .iloc for positional indexing
    if x.iloc[feature] <= 0.5:
        return predict_sample(x, tree["left_branch"])
    else:
        return predict_sample(x, tree["right_branch"])


def predict(X, tree):
    """
    Predict target values y given a decision tree.
    """
    # Initialize an array to store predictions
    predictions = np.zeros(X.shape[0])
    # Iterate over all samples and store predictions
    for i in range(X.shape[0]):
        predictions[i] = predict_sample(X.iloc[i], tree)
    return predictions


def compute_accuracy(y_pred, y_true):
    """
    Compute accuracy given predicted and true target values.
    """
    return np.mean(y_pred == y_true)

def visualize_tree(tree, feature_names, class_names, save_name=None):
    graph = pydot.Dot(graph_type="graph")
    def add_nodes_and_edges(node, parent=None):
        if isinstance(node, dict):  # Check if it's an internal node
            label = feature_names[node["best_feature"]]
            node_id = id(node)  # Unique identifier for the node
            graph_node = pydot.Node(node_id, label=label)

            # Add node to the graph
            graph.add_node(graph_node)

            # Add edge from parent if it exists
            if parent is not None:
                graph.add_edge(pydot.Edge(parent, node_id))

            # Recursive calls for children
            add_nodes_and_edges(node["left_branch"], node_id)
            add_nodes_and_edges(node["right_branch"], node_id)

        else:  # Leaf node
            label = class_names[node]
            graph_node = pydot.Node(id(node), label=label)
            graph.add_node(graph_node)
            if parent is not None:
                graph.add_edge(pydot.Edge(parent, id(node)))

    # Start recursive process
    add_nodes_and_edges(tree)

    # Save or display the graph
    if save_name:
        graph.write_png(save_name)
    return graph


def load_data():
    data = pd.read_csv('./spotify_songs.csv')
    data = data[:29000]
    return data

def separate_data(data):
    X = data.drop(columns=['track_id', 'track_name', 'track_album_id', 'track_artist', 
                           'playlist_name', 'playlist_genre', 'playlist_subgenre', 
                           'key', 'mode', 'track_popularity', 'track_album_name', 
                           'track_album_release_date', 'playlist_id', 'loudness', 
                           'duration_ms', 'valence'])
    y = data['track_popularity']
    return X, y


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def visualize_sklearn_tree(decision_tree, feature_names, class_names, save_name="decision_tree"):
    # Convert feature_names to a list, if it isn't already
    feature_names = list(feature_names)
    
    plt.figure(figsize=(20,10))  # Set the figure size (width, height)
    plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
    plt.savefig(f"{save_name}.svg", format='svg')  # Save as vector graphic
    plt.savefig(f"{save_name}.png", format='png', dpi=300)  # Save as high-resolution PNG
    plt.close()  # Close the plot to free memory


def main():
    # Load data
    file_path = './spotify_songs.csv'
    data = pd.read_csv(file_path)

    # Drop irrelevant or non-informative features
    data = data.drop(columns=['track_id', 'track_name', 'track_album_id', 'track_artist', 
                            'playlist_name', 'playlist_genre', 'playlist_subgenre', 
                            'track_album_name', 'track_album_release_date', 'playlist_id'])

    # Convert 'track_popularity' into three classes: Low, Medium, High
    popularity_bins = [0, 33, 66, 100]  # Adjust bins as needed
    popularity_labels = [0, 1, 2]
    data['track_popularity'] = pd.cut(data['track_popularity'], bins=popularity_bins, labels=popularity_labels, include_lowest=True, right=True)

    # Encode categorical features if any
    # Here, assuming 'key' and 'mode' are the only categorical features left
    label_encoder = LabelEncoder()
    # data['key'] = label_encoder.fit_transform(data['key'])
    # data['mode'] = label_encoder.fit_transform(data['mode'])

    # Separate features and target
    X = data.drop('track_popularity', axis=1)
    y = label_encoder.fit_transform(data['track_popularity'])  # Convert labels to integers

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # sm = SMOTE(random_state=42)
    # X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # Build the decision tree with the balanced dataset
    tree = build_tree(X_train, y_train, max_depth=10, min_size=150)
    # Make predictions
    y_pred = predict(X_test, tree)
    # Compute accuracy
    accuracy = compute_accuracy(y_pred, y_test)
    print("Accuracy:", accuracy * 100)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1:", f1*100)
    visualize_tree(tree, X.columns, np.unique(y), save_name="decision_tree.png")

    # use library to compare
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=50)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy = compute_accuracy(y_pred, y_test)
    print("Accuracy:", accuracy * 100)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1:", f1*100)

    # cross validation
    # scores = cross_val_score(tree, X, y, cv=5)
    # print(scores)

    # Visualize the tree
    class_names_str = [str(cls) for cls in np.unique(y)]
    visualize_sklearn_tree(tree, X.columns.tolist(), class_names_str, save_name="decision_tree_lib.png")

    # visualize_sklearn_tree(tree, X.columns.tolist(), np.unique(y).tolist(), save_name="decision_tree_lib.png")

if __name__ == "__main__":
    main()

