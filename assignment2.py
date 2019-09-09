import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pickle
import copy

def load_batch(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1') # Nxd(3072) (Nx (32x32x3))
        X = np.transpose(dataset['data'] / 255.) # d x N
        mean_X = np.mean(X, axis=1) # mean of each row (each feature mean)
        std_X = np.std(X, axis=1)
        X = X - np.matlib.repmat(mean_X, X.shape[1], 1).T
        X = np.divide(X, np.matlib.repmat(std_X, X.shape[1], 1).T)
        
        y = np.array(dataset['labels'])
        Y = np.transpose(np.eye(X.shape[1], np.max(y) + 1)[y]) # K x N
        return X, Y, y

def load_all(validation_size):
    X_1, Y_1, y_1 = load_batch('data/data_batch_1')
    X_2, Y_2, y_2 = load_batch('data/data_batch_2')
    X_3, Y_3, y_3 = load_batch('data/data_batch_3')
    X_4, Y_4, y_4 = load_batch('data/data_batch_4')
    X_5, Y_5, y_5 = load_batch('data/data_batch_5')
    
    X = np.concatenate((X_1, X_2, X_3, X_4, X_5[:,:-validation_size]), axis=1)
    Y = np.concatenate((Y_1, Y_2, Y_3, Y_4, Y_5[:,:-validation_size]), axis=1)
    y = np.concatenate((y_1, y_2, y_3, y_4, y_5[:-validation_size]))
    
    X_valid = X_5[:,-validation_size:]
    Y_valid = Y_5[:,-validation_size:]
    y_valid = y_5[-validation_size:]
    return X, Y, y, X_valid, Y_valid, y_valid
    
def softmax(s):
    exponent = np.exp(s)
    return np.divide(exponent, np.sum(exponent, axis=0))

def evaluate_classifier(X, layers):
    num_layers = len(layers)
    H = []
    h_prev = X
    
    for i, layer in enumerate(layers):
        if i == num_layers - 1:  # If last layer
            P = softmax(np.dot(layer["W"], h_prev) + layer["b"]) # K x N
            return H, P
        else:
            s = np.dot(layer["W"], h_prev) + layer["b"] # m x N
            h = np.maximum(s, 0) # ReLU; m x N
            H.append(h)
            h_prev = h

def compute_cost(X, Y, layers, lmb):
    H, P = evaluate_classifier(X, layers)
    n = np.sum(np.multiply(Y, P), axis=0)
    cross_entropy = np.sum(-np.log(n))
    
    w_square_sum = 0
    if lmb > 0:
        for layer in layers:
            w_square_sum += np.sum(np.diag(np.dot(layer["W"].T, layer["W"])))
    return (cross_entropy / X.shape[1]) + (lmb * w_square_sum)

def compute_gradients(X, Y, layers, lmb):
    H, P = evaluate_classifier(X, layers)
    G = -(Y - P)
    Nb = X.shape[1] # batch size
    
    W_gradients = []
    b_gradients = []
    for i, layer in reversed(list(enumerate(layers))): # from last to first
        if i > 0:
            grad_W = np.divide(np.dot(G, H[i - 1].T), Nb) + (2 * lmb * layer["W"])
            grad_b = np.divide(np.dot(G, np.ones((Nb, 1))), Nb)
            G = np.dot(layer["W"].T, G)
            G = G * (H[i - 1] > 0).astype(int) # element-wise
            
            W_gradients.append(grad_W)
            b_gradients.append(grad_b)
        else: # first layer
            grad_W = np.divide(np.dot(G, X.T), Nb) + (2 * lmb * layer["W"])
            grad_b = np.divide(np.dot(G, np.ones((Nb, 1))), Nb)
            W_gradients.append(grad_W)
            b_gradients.append(grad_b)
    return W_gradients, b_gradients

    
def compute_gradients_num(X, Y, layers, lmb, h):
    
    grad_W = [np.zeros(layer["W"].shape) for layer in layers]
    grad_b = [np.zeros(layer["W"].shape[0]) for layer in layers]
    c = compute_cost(X, Y, layers, lmb)
    
    for l, layer in enumerate(layers):
        for i in range(len(layer["b"])):
            layers_try = copy.deepcopy(layers)
            layers_try[l]["b"][i] = layers_try[l]["b"][i] + h
            c2 = compute_cost(X, Y, layers_try, lmb)
            grad_b[l][i] = (c2 - c) / h

        W_shape = layer["W"].shape
        for i in range(W_shape[0]):
            for j in range(W_shape[1]):
                layers_try = copy.deepcopy(layers)
                layers_try[l]["W"][i,j] = layers_try[l]["W"][i,j] + h
                c2 = compute_cost(X, Y, layers_try, lmb)
                grad_W[l][i,j] = (c2 - c) / h
        
    return grad_W, grad_b

def compute_accuracy(X, y, layers):
    _, p = evaluate_classifier(X, layers)
    argmax = np.argmax(p, axis=0) # max element index of each column
    diff = argmax - y
    return (diff == 0).sum() / X.shape[1]

def mini_batch_GD(X, Y, GDparams, layers, lmb, validation, calculate_loss=False):
#     print("Training samples: {}".format(X.shape[1]))
#     print("Validation samples: {}".format(validation["X"].shape[1]))
#     print("Training parameters: ", GDparams)
    J_training = []
    J_validation = []
    
    eta_diff = GDparams["eta_max"] - GDparams["eta_min"]
    eta = GDparams["eta_min"]
    t = 0 # step
    l = 0 # cycle
    
    runs_in_epoch = int(X.shape[1] / GDparams["n_batch"])
    # for epoch in range(GDparams["epochs"]):
    while l < GDparams["max_cycles"]:
        for j in range(1, runs_in_epoch):
            j_start = (j - 1) * GDparams["n_batch"]
            j_end = j * GDparams["n_batch"]
            
            X_batch = X[:, j_start:j_end]
            Y_batch = Y[:, j_start:j_end]
            
            grad_W, grad_b = compute_gradients(X_batch, Y_batch, layers, lmb)
            
            for i, layer in enumerate(layers):
                layer["W"] = layer["W"] - (eta * grad_W[-1 - i])
                layer["b"] = layer["b"] - (eta * grad_b[-1 - i])
        
            if calculate_loss and t % 100 == 0:
                J_training.append(compute_cost(X, Y, layers, lmb))
                J_validation.append(compute_cost(validation["X"], validation["Y"], layers, lmb))
                print("Step {}, training loss: {}".format(t, J_training[-1]))
                
            t += 1 # next update step
            if t % (2 * GDparams["n_s"]) == 0:
                l += 1 # next cycle
                if l == GDparams["max_cycles"]:
                    break
#                 print("Entering cycle {}, t: {}, eta: {}".format(l, t, eta))
            if t <= (2*l + 1) * GDparams["n_s"]:
                eta = GDparams["eta_min"] + (eta_diff * ((t - (2 * l * GDparams["n_s"])) / GDparams["n_s"]))
            else:
                eta = GDparams["eta_max"] - (eta_diff * (t - ((2*l + 1) * GDparams["n_s"])) / GDparams["n_s"])

    if calculate_loss:
        return layers, J_training, J_validation
    else:
        return layers

# X, Y, y = load_batch('data/data_batch_1')
# X_valid, Y_valid, y_valid = load_batch('data/data_batch_2')
X, Y, y, X_valid, Y_valid, y_valid = load_all(validation_size=1000)
X_test, Y_test, y_test = load_batch('data/test_batch')

d = X.shape[0]
N = X.shape[1]
K = Y.shape[0]

# X: d x N
# Y: K x N

m = 50 # hidden units

std_dev_1 = 1 / np.sqrt(d)
std_dev_2 = 1 / np.sqrt(m)

def init_network(dimensions=d):
    layers = [
        {
            "W": std_dev_1 * np.random.randn(m, dimensions),
            "b": np.zeros((m, 1)),
        },
        {
            "W": std_dev_2 * np.random.randn(K, m),
            "b": np.zeros((K, 1)),
        },
    ]
    return layers


lmb = 0.01  # lambda
GDparams = {
    "n_batch": 100,
    "eta_min": 1e-5,
    "eta_max": 1e-1,
    "n_s": 2 * np.floor(X.shape[1] / 100),,
    "epochs": 10,
    "max_cycles": 3,
}
validation = {
    "X": X_valid,
    "Y": Y_valid,
}

layers_trained = mini_batch_GD(X, Y, GDparams, layers, lmb, validation)
test_cost = compute_cost(X_test, Y_test, layers_trained, lmb)
test_acc = compute_accuracy(X_test, y_test, layers_trained)

print("\nTest cost: {}".format(test_cost))
print("Test accuracy: {}".format(test_acc))


###### Checking gradients ########
feature_dims = 20
samples = 1

layers_g = init_network(dimensions=feature_dims)
X_g = X[0:feature_dims, 0:samples] #d X N
Y_g = Y[:, 0:samples] #K x N

grad_W, grad_b = compute_gradients(X_g, Y_g, layers_g, lmb=0)
ngrad_w, ngrad_b = compute_gradients_num(X_g, Y_g, layers_g, lmb=0, h=1e-5)
print("\ngrad b:")

print(np.abs(grad_b[1].T - ngrad_b[0]) / np.maximum(1e-5, np.abs(grad_b[1].T) + np.abs(ngrad_b[0])))
print(np.abs(grad_b[0].T - ngrad_b[1]) / np.maximum(1e-5, np.abs(grad_b[0].T) + np.abs(ngrad_b[1])))

print("\ngrad W:")
print(np.abs(grad_W[1] - ngrad_w[0]) / np.maximum(1e-5, np.abs(grad_W[1]) + np.abs(ngrad_w[0])))
print(np.abs(grad_W[0] - ngrad_w[1]) / np.maximum(1e-5, np.abs(grad_W[0]) + np.abs(ngrad_w[1])))
#######################################


# Searching for lambda (coarse)
l_min = -7
l_max = -0.5

ls = np.sort([np.random.uniform(l_min, l_max) for i in range(40)])
for l in ls:
    lmb = np.power(10, l)
    layers = init_network()
    layers_trained = mini_batch_GD(X, Y, GDparams, layers, lmb, validation)
    test_cost = compute_cost(X_test, Y_test, layers_trained, lmb)
    test_acc = compute_accuracy(X_test, y_test, layers_trained)
    print("Lambda: {} (l: {}), test cost: {}, test accuracy: {}".format(lmb, l, test_cost, test_acc))


# Searching for lambda (fine)
l_min = -3.2
l_max = -2.2

ls = np.sort([np.random.uniform(l_min, l_max) for i in range(40)])
for l in ls:
    lmb = np.power(10, l)
    layers = init_network()
    layers_trained = mini_batch_GD(X, Y, GDparams, layers, lmb, validation)
    test_cost = compute_cost(X_test, Y_test, layers_trained, lmb)
    test_acc = compute_accuracy(X_test, y_test, layers_trained)
    print("Lambda: {} (l: {}), test cost: {}, test accuracy: {}".format(lmb, l, test_cost, test_acc))


# Best
lmb = np.power(10, -2.325238114712273)
layers = init_network()
layers_trained, J_training, J_validation = mini_batch_GD(X, Y, GDparams, layers, lmb, validation, calculate_loss=True)
test_cost = compute_cost(X_test, Y_test, layers_trained, lmb)
test_acc = compute_accuracy(X_test, y_test, layers_trained)

plt.figure(0)
plt.ylabel("Cost")
plt.xlabel("Step")
plt.plot([step*100 for step in range(len(J_training))], J_training, label="Training")
plt.plot([step*100 for step in range(len(J_validation))], J_validation, label="Validation")
plt.legend()
plt.show()

print("Test cost: ", test_cost)
print("Test accuracy: ", test_acc)