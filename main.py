import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import issparse


# load data using numpy
def load_data():
    data = pd.read_csv('./spotify_songs.csv')
    # use only 100 songs
    data = data[:100]
    # print(data)
    return data

def read_data():
    data = pd.read_csv('./spotify_songs.csv')
    # use only 15000 songs
    data = data[:1000]
    if 'track_popularity' not in data.columns:
        print("Column 'track_popularity' does not exist in the dataset.")
        exit(1)

    # print name of columns
    # print(data.columns)
    return data

def X_train_data(data):
    # drop everything but ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
    X = data.drop(columns=['track_id', 'track_name','track_album_id' ,'track_artist', 'playlist_name', 'playlist_genre', 'playlist_subgenre', 'key', 'mode', 'track_popularity', 'track_album_name', 'track_album_release_date', 'playlist_id'])
    print(X)
    return X

def y_train_data(data):
    y = data['track_popularity']  # Target
    return y


def sigmoid(z):
    # Clip z to prevent overflow or underflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z)).astype(np.float32)


def compute_cost(X, y, w, b, lambda_=1):
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    w = w.astype(np.float32)
    b = b.astype(np.float32)

    m, n = X.shape
    
    ### START CODE HERE ###
    Z = np.dot(X,w) + b
    # print(Z)
    F_wb = sigmoid(Z)
    # print(F_wb)
    # total_cost = np.sum((F_wb - y) ** 2) / (2 * m)
    loss_i = (-y*np.log(F_wb + 0.00001)) - (1-y)*np.log(1-F_wb+0.00001) 
    total_cost = np.sum(loss_i) / m
    ### END CODE HERE ### 

    return total_cost

def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    Z = X.dot(w) + b  # Shape of Z should be (m, 1)
    F_wb = sigmoid(Z)  # Shape of F_wb should be (m, 1)
    F_wb = F_wb.astype(np.float32)
    # Reshape y to (m, 1) if it's not already
    dj_dw = np.dot(X.T, (F_wb - y)) / m
    dj_db = np.sum(F_wb - y) / m
    return dj_db.astype(np.float32), dj_dw.astype(np.float32)


def gradient_descent(X, y, w_in, b_in,alpha, num_iters, lambda_):
    # number of training examples
    m = len(X)

    # print(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient(X, y, w_in, b_in, lambda_)   

        # print (f"Gradient: {dj_dw} {dj_db} ")

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db            

        # print(f"Parameters: {w_in} {b_in} ")  
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  compute_cost(X, y, w_in, b_in, lambda_)
            # print(cost)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing

def predict(X, w, b):
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    
    Z = np.dot(X,w) + b
    F_wb = sigmoid(Z)
    p = np.where(F_wb > 0.5, 1, 0)

        
    ### END CODE HERE ### 
    return p

def main():
    # data = read_data()
    
    data = load_data()
    # data_test = data_test[:100]

    # print("X_train_data:\n",X_train_data(data))
    # print("y_train_data:\n",y_train_data(data))

    X = X_train_data(data)
    y = y_train_data(data)
    
    # exit(0)
    # Preprocess categorical and numerical features
    # categorical_features = ['playlist_genre', 'playlist_subgenre', 'key', 'mode']  # list all categorical features
    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']  # list all numerical features


    # print("Columns in DataFrame:", data.columns)

    # Separating the target variable
    # X = data.drop(columns=['track_popularity'])  # Features
    # print(X)
    # print first 10 rows of X
    # print(X.head(10))
    # y = data['track_popularity']  # Target

    # check if all columns in categorical_features exist in X
    # for col in categorical_features:
    #     if col not in X.columns:
    #         print(f"Column '{col}' does not exist in the dataset.")
    #         exit(1)

    # check if all columns in numerical_features exist in X
    for col in numerical_features:
        if col not in X.columns:
            print(f"Column '{col}' does not exist in the dataset.")
            exit(1)

    # exit(0)
    # Debugging: Print columns of X and feature lists
    # print("Columns in X after dropping 'track_popularity':", X.columns)
    # print("Categorical features:", categorical_features)
    # print("Numerical features:", numerical_features)

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', StandardScaler(), numerical_features),
    #         ('cat', OneHotEncoder(sparse=True), categorical_features)])


    # Apply preprocessing
    # try:
    #     X = preprocessor.fit_transform(X)
    #     X = X.astype(np.float32)  # Convert to float32 after preprocessing
    #     print("X after preprocessing:\n", X)
    # except ValueError as e:
    #     print("Error in preprocessing:", e)
    #     print('\n')
    #     return
    # y = np.array(y).ravel()  # Convert y to a 1D array (vector)

    # Convert to dense matrix if memory permits
    # print(X[1:])
    # print(type(X))
    # print(type(X['danceability'][1]))
    # if isinstance(X, np.ndarray):
    #     print("X is already a dense matrix.\n")
    # else:
    #     print("Converting X to a dense matrix.\n")
    #     # ignore the first row of X since it's the header
    #     X = X.to_xarray()

    # exit(0)
    # Split data into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X)
    # Initialize weights and bias
    w = np.zeros((X.shape[1], 1), dtype=np.float32)
    b = np.float32(0)

    # Train the model
    alpha = 0.01
    num_iters = 1000

    w, b, J_history, w_history = gradient_descent(X, y, w, b, alpha, num_iters, None)

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
