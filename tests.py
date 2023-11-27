from main import *

def load_data():
    data = pd.read_csv('./spotify_songs.csv')
    print(type(data))
    data = data[:100]
    return data

def test_sigmoid():
    print(sigmoid(0))
    print(sigmoid(1))
    print(sigmoid(np.array([-1, 0, 1, 2])))
    return True

def test_compute_cost():
    X, y = load_test_data()
    m,n = X.shape
    w = np.zeros(n)
    b = 0
    cost = compute_cost(X,y,w,b)
    print('Cost at initial w,b:',cost)
    return True

def test_compute_gradient():
    X, y = load_test_data()
    m,n = X.shape
    w = np.zeros(n)
    b = 0
    dj_dw, dj_db = compute_gradient(X,y,w,b)
    print(f'dj_db at initial w (zeros):{dj_db}' )
    print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}' )
    return True

def test_gradient_descent():
    np.random.seed(1)
    X_train, y_train = load_test_data()
    y_train = categorize_popularity(y_train)

    initial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
    print("initial_w:",initial_w)
    initial_b = -8

    # w,b, J_history, w_history = gradient_descent(X_train ,y_train, initial_w, initial_b, 
    #                                 compute_cost, compute_gradient, alpha, iterations, 0)

    num_classes = 2
    alpha = 0.00001
    num_iters = 10000

    models = train_multiclass(X_train, y_train, num_classes, alpha, num_iters)
    predictions = predict_multiclass(X_train, models)

    print("Accuracy:", accuracy_score(y_train, predictions)*100, "%")
    print("Confusion Matrix:\n", confusion_matrix(y_train, predictions))

    return True

def tests():
    # print("Running tests...\n_____________________________\n_____________________________")
    # print("_____________________________\nTesting sigmoid function...\n_____________________________")
    # test_sigmoid()
    # print("_____________________________\nTesting compute_cost function...\n_____________________________")
    # test_compute_cost()
    # print("_____________________________\nTesting compute_gradient function...\n_____________________________")
    # test_compute_gradient()
    print("_____________________________\nTesting gradient_descent function...\n_____________________________")
    test_gradient_descent()
    return True

if __name__ == "__main__":
    tests()