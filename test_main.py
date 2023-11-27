import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import issparse
import statsmodels.api as sm


# load data using numpy
def load_data():
    data = pd.read_csv('./spotify_songs.csv')
    print(type(data))
    # use only 100 songs
    data = data[:100]
    # print(data)
    return data

def read_data():
    data = pd.read_csv('./spotify_songs.csv')
    type(data)
    # use only 15000 songs
    data = data[:1000]
    if 'track_popularity' not in data.columns:
        print("Column 'track_popularity' does not exist in the dataset.")
        exit(1)

    # print name of columns
    # print(data.columns)
    return data

def X_train_data(data):
    # Drop columns that are not features
    X = data.drop(columns=['track_id', 'track_name', 'track_album_id', 'track_artist', 'playlist_name',
                           'playlist_genre', 'playlist_subgenre', 'key', 'mode', 'track_popularity',
                           'track_album_name', 'track_album_release_date', 'playlist_id'])
    return X.astype(np.float32)  # Convert to float32

def y_train_data(data):
    return data['track_popularity'].astype(np.float32)  # Convert to float32


def sigmoid(z):
    return 1 / (1 + np.exp(-z)).astype(np.float32)


def compute_cost(X, y, w, b):
    m, n = X.shape
    Z = np.dot(X, w) + b
    F_wb = sigmoid(Z)
    # print(Z)
    # print(F_wb)
    # exit(1)

    # Ensure F_wb is a 2D array with shape (m, 1)
    F_wb = F_wb.reshape(-1, 1)

    # Convert y to a 2D array with shape (m, 1) if it's not already
    if isinstance(y, pd.Series):
        y = y.values.reshape(-1, 1)
    elif isinstance(y, np.ndarray) and y.ndim == 1:
        y = y.reshape(-1, 1)


    loss_i = (-y*np.log(F_wb + 1e-5)) - (1-y)*np.log(1-F_wb+1e-5)
    total_cost = np.sum(loss_i) / m
    return total_cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    Z = np.dot(X, w) + b
    F_wb = sigmoid(Z)

    dj_db = 0
    dj_dw = np.zeros(w.shape)

    if isinstance(y, pd.Series):
        y = y.values.reshape(-1, 1)
    elif isinstance(y, np.ndarray) and y.ndim == 1:
        y = y.reshape(-1, 1)


    dj_dw = np.dot(X.T, (F_wb - y)) / m
    dj_db = np.sum(F_wb - y) / m
    return dj_db, dj_dw


def gradient_descent(X, y, w, b, alpha, num_iters):
    J_history = []
    w_history = []

    m = len(X)

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b)

        # print("Max Z value:", np.max(np.dot(X, w) + b))
        # print("Min F_wb value:", np.min(sigmoid(np.dot(X, w) + b)))
        # print("Max F_wb value:", np.max(sigmoid(np.dot(X, w) + b)))



        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history

def predict(X, w, b):
    Z = np.dot(X, w) + b
    F_wb = sigmoid(Z)
    return np.where(F_wb > 0.5, 1, 0)


def test_data():
    test_X = np.random.rand(100, 10).astype(np.float32)
    test_w = np.random.rand(10, 1).astype(np.float32)
    test_b = np.float32(0)
    try:
        test_Z = np.dot(test_X, test_w) + test_b
        print("test_Z:\n", test_Z)
        test_F_wb = sigmoid(test_Z)
        print("test_F_wb:\n", test_F_wb)
        test_cost = compute_cost(test_X, test_F_wb, test_w, test_b)
        print("test_cost:\n", test_cost)
        test_dj_db, test_dj_dw = compute_gradient(test_X, test_F_wb, test_w, test_b)
        print("test_dj_db:\n", test_dj_db)
        print("test_dj_dw:\n", test_dj_dw)
        
    except ValueError as e:
        print("Error during test:", e)


def main():
    data = load_data()

    X = X_train_data(data)
    y = y_train_data(data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train = sm.add_constant(X_train)



    print(X_train)

    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
    for col in numerical_features:
        if col not in X.columns:
            print(f"Column '{col}' does not exist in the dataset.")
            return

    w = np.random.randn(X_train.shape[1], 1) * 0.01
    b = np.float32(0)

    alpha = 0.01
    num_iters = 1000

    w, b, J_history, w_history = gradient_descent(X_train, y_train, w, b, alpha, num_iters)

    predictions = predict(X, w, b)

    # Convert predictions and y_test to the same shape for evaluation
    # predictions = predictions.ravel()
    y_test = y_test.ravel()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')  # or 'micro' or 'weighted'
    recall = recall_score(y_test, predictions, average='macro')  # or 'micro' or 'weighted'
    f1 = f1_score(y_test, predictions, average='macro')  # or 'micro' or 'weighted'
    conf_matrix = confusion_matrix(y_test, predictions)

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == '__main__':
    main()
