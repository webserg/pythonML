import numpy as np

def L_model_backward_test_case_2hidden():
    np.random.seed(7)
    AL = np.random.rand(1, 5)
    Y = np.array([[0., 1., 0., 1., 1.]])

    A0 = np.random.randn(4,5)
    W1 = np.random.randn(7,4)
    b1 = np.random.randn(7,1)
    Z1 = np.random.randn(7,5)
    linear_cache_activation_1 = ((A0, W1, b1), Z1)

    A1 = np.random.randn(7,5)
    W2 = np.random.randn(3,7)
    b2 = np.random.randn(3,1)
    Z2 = np.random.randn(3,5)
    linear_cache_activation_2 = ((A1, W2, b2), Z2)

    A2 = np.random.randn(3,5)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
    Z3 = np.random.randn(1,5)
    linear_cache_activation_3 = ((A2, W3, b3), Z3)
    
    caches = (linear_cache_activation_1, linear_cache_activation_2, linear_cache_activation_3)

    return AL, Y, caches

def print_grads_2hidden(grads):
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dA1 = "+ str(grads["dA2"]))
    print ("dW2 = "+ str(grads["dW2"]))
    print ("db2 = "+ str(grads["db2"]))
    print ("dA2 = "+ str(grads["dA3"]))