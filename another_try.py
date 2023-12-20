from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    data = pd.read_csv('./spotify_songs.csv')
    data = data[:6000]
    return data

def separate_data(data):
    X = data.drop(columns=['track_id', 'track_name', 'track_album_id', 'track_artist', 
                           'playlist_name', 'playlist_genre', 'playlist_subgenre', 
                           'key', 'mode', 'track_popularity', 'track_album_name', 
                           'track_album_release_date', 'playlist_id', 'loudness', 
                           'duration_ms', 'valence'])
    y = data['track_popularity']
    return X, y

def init_weights(n_features):
    w = np.zeros((n_features, 1))  # Ensure w is a column vector
    b = 0.
    return w, b

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def logloss(y_true, y_pred):
    # Convert y_true to numpy array if it's a pandas Series
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()

    # Convert y_pred to numpy array if it's a list
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    # Now you can safely reshape
    y_pred = y_pred.reshape(y_true.shape)

    # Clipping to avoid division by zero in log
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    y_true = np.clip(y_true, 1e-15, 1 - 1e-15)

    # Calculate log loss
    log_loss = -1 * np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return log_loss

def compute_class_weights(y):
    """
    Compute class weights to handle class imbalance.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, class_weights))


def gradient_dw(X, y, w, b, class_weights, alpha):
    N = len(X)
    weighted_sum = np.zeros(w.shape)

    for i in range(N):
        xi = X[i, :].reshape(1, -1)
        yi = y[i]
        weight = class_weights[yi]
        weighted_sum += weight * xi.T * (sigmoid(xi @ w + b) - yi)

    dw = weighted_sum / N + alpha * w  # Adding regularization term
    return dw


def gradient_db(X, y, w, b, class_weights):
    N = len(X)
    weighted_sum = 0

    for i in range(N):
        xi = X[i, :].reshape(1, -1)
        yi = y[i]
        weight = class_weights[yi]
        weighted_sum += weight * (sigmoid(xi @ w + b) - yi)

    db = np.mean(weighted_sum)
    return db

def train(X_train, y_train, X_test, y_test, epochs, alpha, eta0):
    w, b = init_weights(X_train.shape[1])
    class_weights = compute_class_weights(y_train)

    for epoch in range(epochs):
        # Adjust the gradient calculation to include class weights and regularization
        dw = gradient_dw(X_train, y_train, w, b, class_weights, alpha)
        db = gradient_db(X_train, y_train, w, b, class_weights)

        # Update weights and bias
        w -= eta0 * dw
        b -= eta0 * db

        if epoch % 100 == 0:
            # Print loss for monitoring
            train_loss = logloss(y_train, sigmoid(X_train @ w + b))
            test_loss = logloss(y_test, sigmoid(X_test @ w + b))
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}")

    return w, b


def categorize_y(y, thresholds):
    categories = np.zeros_like(y, dtype=int)
    categories[y > thresholds[1]] = 2
    categories[(y > thresholds[0]) & (y <= thresholds[1])] = 1

    print(f"Class distribution: {np.bincount(categories)}")
    return categories

def train_ovr(X, y, num_classes, epochs, alpha, eta0):
    models = []
    for class_label in range(num_classes):
        y_binary = (y == class_label).astype(int)
        w, b = train(X, y_binary, X, y_binary, epochs, alpha, eta0)
        models.append((w, b))
    return models

def predict_ovr(X, models):
    # Initialize an array to store predictions
    predictions = np.zeros((X.shape[0], len(models)))

    # Iterate over each model and store predictions
    for i, (w, b) in enumerate(models):
        predictions[:, i] = sigmoid(X @ w + b).flatten()

    # Return the index of the class with the highest probability for each sample
    return np.argmax(predictions, axis=1)

def predict_proba_ovr(X, models):
    """
    Predict probabilities for each class using the trained models.
    """
    # Predicted probabilities for each class
    proba = np.array([sigmoid(X @ w + b).flatten() for w, b in models])
    return proba.T  # Transpose so that shape becomes (n_samples, n_classes)


def main():
    data = load_data()
    X, y = separate_data(data)
    y = categorize_y(y, thresholds=[40, 61])  # Set appropriate thresholds
    print(f"Class distribution in training data: {np.bincount(y)}")
    # exit(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Class distribution in training data: {np.bincount(y_train)}")
    models = train_ovr(X_train_scaled, y_train, num_classes=3, epochs=500, alpha=0.01, eta0=0.01)

    y_pred = predict_ovr(X_test_scaled, models)

    # Ensure y_test is a 1D numpy array
    y_test = y_test.flatten() if y_test.ndim > 1 else y_test

    accuracy = np.mean(y_pred == y_train)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Class distribution in test data: {np.bincount(y_pred)}")
    print(f"Class distribution in test data: {np.bincount(y_test)}")


    # Additional metrics
    from sklearn.metrics import confusion_matrix , classification_report
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

    return True

if __name__ == "__main__":
    main()






