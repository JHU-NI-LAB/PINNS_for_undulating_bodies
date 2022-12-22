"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np

def tf_session():
    # tf session
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.compat.v1.Session(config=config)
    
    # init
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    
    return sess

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(input_tensor=tf.square(pred - exact))/tf.reduce_mean(input_tensor=tf.square(exact - tf.reduce_mean(input_tensor=exact))))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(input_tensor=tf.square(pred - exact))

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y) #creates tensor of all ones that same shape as y
    G = tf.gradients(ys=Y, xs=x, grad_ys=dummy)[0]
    Y_x = tf.gradients(ys=G, xs=dummy)[0]
    return Y_x

class neural_net(object):
    def __init__(self, *inputs, layers):

        self.layers = layers
        self.num_layers = len(self.layers)
        
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            #combine inputs into one matrix X: col1 is input 1, col2 is input 2, etc.
            X = np.concatenate(inputs, 1)
            #Compute the average of each column
            self.X_mean = X.mean(0, keepdims=True)
            # Compute the STDEV of each column
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []
        #set up layers of neural network
        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l] #number of inputs supplied
            out_dim = self.layers[l+1] #number of at the next layer (i.e neurons at next layer)
            W = np.random.normal(size=[in_dim, out_dim]) #assign weigths according to a normal distribution
            #each point on the current layer has to connect to one point on the next layer
            b = np.zeros([1, out_dim]) #biases zero
            g = np.ones([1, out_dim]) #gammas one
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))
            #for each layer append results to weights, biases, and gammas

    def __call__(self, *inputs):
                
        H = (tf.concat(inputs, 1) - self.X_mean)/self.X_std
    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(tensor=W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.sigmoid(H)
                
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y

def Navier_Stokes_2D(u, v, p, t, x, y, Rey):
    Y = tf.concat([u, v, p], 1)  # merge into one matrix along the columns

    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)

    u = Y[:, 0:1]
    v = Y[:, 1:2]
    p = Y[:, 2:3]

    u_t = Y_t[:, 0:1]
    v_t = Y_t[:, 1:2]

    u_x = Y_x[:, 0:1]
    v_x = Y_x[:, 1:2]
    p_x = Y_x[:, 2:3]

    u_y = Y_y[:, 0:1]
    v_y = Y_y[:, 1:2]
    p_y = Y_y[:, 2:3]

    u_xx = Y_xx[:, 0:1]
    v_xx = Y_xx[:, 1:2]

    u_yy = Y_yy[:, 0:1]
    v_yy = Y_yy[:, 1:2]

    e1 = u_t + (u * u_x + v * u_y) + p_x - (1.0 / Rey) * (u_xx + u_yy)
    e2 = v_t + (u * v_x + v * v_y) + p_y - (1.0 / Rey) * (v_xx + v_yy)
    e3 = u_x + v_y

    return e1, e2, e3

def Gradient_Velocity_2D(u, v, x, y):
    
    Y = tf.concat([u, v], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    
    return [u_x, v_x, u_y, v_y]

