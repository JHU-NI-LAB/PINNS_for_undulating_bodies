"""
The original code used to implement PINNs was written by Maziar Raissi (Brown University)
Modifications have been made by Michael A. Calicchia (Johns Hopkins University) to apply PINNs to moving bodies

"""
#This PINN can be used to reconstruct the velocity and pressure field around two-dimensional, undulating bodies#

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import scipy.io
import time
import sys

from utilities import neural_net, Navier_Stokes_2D, Gradient_Velocity_2D, \
                      tf_session, mean_squared_error, relative_error

class HFM(object):
    # notational conventions
        # _tf: placeholders for input/output data and points used to regress the equations
        # _pred: output of neural network
        # _eqns: points used to regress the equations
        # _data: input velocity data
        # _body: denotes points on the surface of the body
        # _bounds_p: external boundary points where pressure boundary condition is enforced
        # _bounds_v: external boundary points where velocity boundary condition is enforced

    def __init__(self, t_data, x_data, y_data, u_data, v_data,
                       t_sec, x_sec, y_sec, u_sec, v_sec,
                       t_eqns, x_eqns, y_eqns,
                       t_bounds_v, x_bounds_v, y_bounds_v, u_bounds, v_bounds,
                       t_bounds_p, x_bounds_p, y_bounds_p, p_bounds,
                       t_body, x_body, y_body, vn_body, nx_body, ny_body,
                       layers, batch_size, save_path,
                       Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size

        # flow properties
        self.Rey = Rey

        # store data
        [self.t_data, self.x_data, self.y_data, self.u_data, self.v_data] = [t_data, x_data, y_data, u_data, v_data]
        [self.t_sec, self.x_sec, self.y_sec,  self.u_sec, self.v_sec] = [t_sec, x_sec, y_sec, u_sec, v_sec]
        [self.t_eqns, self.x_eqns, self.y_eqns] = [t_eqns, x_eqns, y_eqns]
        [self.t_bounds_v, self.x_bounds_v, self.y_bounds_v, self.u_bounds, self.v_bounds] = [t_bounds_v, x_bounds_v, y_bounds_v, u_bounds, v_bounds]
        [self.t_bounds_p, self.x_bounds_p, self.y_bounds_p, self.p_bounds] = [t_bounds_p, x_bounds_p, y_bounds_p, p_bounds]
        [self.t_body, self.x_body, self.y_body, self.vn_body, self.nx_body, self.ny_body] = [t_body, x_body, y_body, vn_body, nx_body, ny_body]

        #These placeholders allow us to create our computation graph without needing the data. We then feed data into the graph through these placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.u_data_tf, self.v_data_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_sec_tf, self.x_sec_tf, self.y_sec_tf, self.u_sec_tf, self.v_sec_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.t_bounds_v_tf, self.x_bounds_v_tf, self.y_bounds_v_tf, self.u_bounds_tf, self.v_bounds_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_bounds_p_tf, self.x_bounds_p_tf, self.y_bounds_p_tf, self.p_bounds_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.t_body_tf, self.x_body_tf, self.y_body_tf, self.vn_body_tf, self.nx_body_tf, self.ny_body_tf] = [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(6)]

        # initialize the neural network
        self.net_cuvp = neural_net(self.t_data, self.x_data, self.y_data, layers = self.layers)

        #obtain PINN predictions at data x,y,t coordinates
        [self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_cuvp(self.t_data_tf,
                                           self.x_data_tf,
                                           self.y_data_tf)

        # obtain PINN predictions at data x,y,t coordinates
        [self.u_sec_pred,
         self.v_sec_pred,
         self.p_sec_pred] = self.net_cuvp(self.t_sec_tf,
                                          self.x_sec_tf,
                                          self.y_sec_tf)

        #obtain PINN predictions at external boundaries (velocities)
        [self.u_bounds_pred,
         self.v_bounds_pred,
         _] = self.net_cuvp(self.t_bounds_v_tf,
                            self.x_bounds_v_tf,
                            self.y_bounds_v_tf)

        #obtain PINN predictions at external boundaries (pressure)
        [_,
         _,
         self.p_bounds_pred] = self.net_cuvp(self.t_bounds_p_tf,
                                             self.x_bounds_p_tf,
                                             self.y_bounds_p_tf)

        #obtain PINN predictions at internal boundaries (body points)
        [self.u_body_pred,
         self.v_body_pred,
         _]= self.net_cuvp(self.t_body_tf,
                            self.x_body_tf,
                            self.y_body_tf)


        #obtain PINN predictions at x,y,t where NSE will be enforced
        [self.u_eqns_pred,
         self.v_eqns_pred,
         self.p_eqns_pred] = self.net_cuvp(self.t_eqns_tf,
                                           self.x_eqns_tf,
                                           self.y_eqns_tf)

        #Compute residuals of NSE
        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         _] = Navier_Stokes_2D(self.u_eqns_pred,
                               self.v_eqns_pred,
                               self.p_eqns_pred,
                               self.t_eqns_tf,
                               self.x_eqns_tf,
                               self.y_eqns_tf,
                               self.Rey)

        [self.e1_eqns_sec_pred,
         self.e2_eqns_sec_pred,
         _] = Navier_Stokes_2D(self.u_sec_pred,
                               self.v_sec_pred,
                               self.p_sec_pred,
                               self.t_sec_tf,
                               self.x_sec_tf,
                               self.y_sec_tf,
                               self.Rey)

        # gradients
        [self.u_x_eqns_pred,
         self.v_x_eqns_pred,
         self.u_y_eqns_pred,
         self.v_y_eqns_pred] = Gradient_Velocity_2D(self.u_eqns_pred,
                                                    self.v_eqns_pred,
                                                    self.x_eqns_tf,
                                                    self.y_eqns_tf)

        # compute loss terms
        self.v_norm = tf.multiply(self.u_body_pred,self.nx_body_tf) + tf.multiply(self.v_body_pred,self.ny_body_tf)
        self.loss_vn_body = mean_squared_error(self.v_norm, self.vn_body_tf)

        self.loss_u = mean_squared_error(self.u_data_pred, self.u_data_tf) + mean_squared_error(self.u_sec_pred, self.u_sec_tf)
        self.loss_v = mean_squared_error(self.v_data_pred, self.v_data_tf) + mean_squared_error(self.v_sec_pred, self.v_sec_tf)
        self.loss_u_bounds = mean_squared_error(self.u_bounds_pred, self.u_bounds_tf)
        self.loss_v_bounds = mean_squared_error(self.v_bounds_pred, self.v_bounds_tf)
        self.loss_p_bounds = mean_squared_error(self.p_bounds_pred, self.p_bounds_tf)
        self.loss_e1 = mean_squared_error(self.e1_eqns_pred, 0.0)
        self.loss_e2 = mean_squared_error(self.e2_eqns_pred, 0.0)

        self.loss = 100*(self.loss_u + self.loss_v + b[0] * self.loss_u_bounds + b[0] * self.loss_v_bounds +
                           b[1] * self.loss_p_bounds + self.loss_vn_body) + (self.loss_e1 + self.loss_e2)

        # optimizers
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()

    def train(self, imax, total_time, learning_rate):

        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
        N_body = self.t_body.shape[0]
        N_bounds_p = self.t_bounds_p.shape[0]
        N_bounds_v = self.t_bounds_v.shape[0]
        N_sec = self.t_sec.shape[0]

        start_time = time.time()
        running_time = 0
        it = 0
        i = 0
        j = int(imax/10)
        loss_value = np.zeros([1, j], dtype='f')
        loss_u = np.zeros([1, j], dtype='f')
        loss_v = np.zeros([1, j], dtype='f')
        loss_u_bounds = np.zeros([1, j], dtype='f')
        loss_v_bounds = np.zeros([1, j], dtype='f')
        loss_vn_body = np.zeros([1, j], dtype='f')
        loss_p_bounds = np.zeros([1, j], dtype='f')
        loss_e1 = np.zeros([1, j], dtype='f')
        loss_e2 = np.zeros([1, j], dtype='f')

        while running_time < total_time:

            idx_data = np.random.choice(N_data, min(5000, N_data))
            idx_eqns = np.random.choice(N_eqns, min(self.batch_size, N_data))
            idx_body = np.random.choice(N_body, min(self.batch_size, N_body))
            idx_bounds_v = np.random.choice(N_bounds_v, min(self.batch_size, N_bounds_v))
            idx_bounds_p = np.random.choice(N_bounds_p, min(self.batch_size, N_bounds_p))
            idx_sec = np.random.choice(N_sec, min(5000, N_sec))

            (t_data_batch,
             x_data_batch,
             y_data_batch,
             u_data_batch,
             v_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.u_data[idx_data,:],
                              self.v_data[idx_data,:])

            (t_sec_batch,
             x_sec_batch,
             y_sec_batch,
             u_sec_batch,
             v_sec_batch) = (self.t_sec[idx_sec, :],
                             self.x_sec[idx_sec, :],
                             self.y_sec[idx_sec, :],
                             self.u_sec[idx_sec, :],
                             self.v_sec[idx_sec, :])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch) = (self.t_eqns[idx_eqns,:],
                              self.x_eqns[idx_eqns,:],
                              self.y_eqns[idx_eqns,:])

            (t_body_batch,
             x_body_batch,
             y_body_batch,
             vn_body_batch,
             nx_body_batch,
             ny_body_batch) = (self.t_body[idx_body,:],
                               self.x_body[idx_body,:],
                               self.y_body[idx_body,:],
                               self.vn_body[idx_body,:],
                               self.nx_body[idx_body,:],
                               self.ny_body[idx_body,:])

            (t_bounds_v_batch,
             x_bounds_v_batch,
             y_bounds_v_batch,
             u_bounds_batch,
             v_bounds_batch) = (self.t_bounds_v[idx_bounds_v, :],
                                self.x_bounds_v[idx_bounds_v, :],
                                self.y_bounds_v[idx_bounds_v, :],
                                self.u_bounds[idx_bounds_v, :],
                                self.v_bounds[idx_bounds_v, :])

            (t_bounds_p_batch,
             x_bounds_p_batch,
             y_bounds_p_batch,
             p_bounds_batch) = (self.t_bounds_p[idx_bounds_p, :],
                                  self.x_bounds_p[idx_bounds_p, :],
                                  self.y_bounds_p[idx_bounds_p, :],
                                  self.p_bounds[idx_bounds_p, :])

# When I want t_data_tf return t_data_batch
            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.u_data_tf: u_data_batch,
                       self.v_data_tf: v_data_batch,
                       self.t_sec_tf: t_sec_batch,
                       self.x_sec_tf: x_sec_batch,
                       self.y_sec_tf: y_sec_batch,
                       self.u_sec_tf: u_sec_batch,
                       self.v_sec_tf: v_sec_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.t_bounds_v_tf: self.t_bounds_v,
                       self.x_bounds_v_tf: self.x_bounds_v,
                       self.y_bounds_v_tf: self.y_bounds_v,
                       self.u_bounds_tf: self.u_bounds,
                       self.v_bounds_tf: self.v_bounds,
                       self.x_bounds_p_tf: self.x_bounds_p,
                       self.y_bounds_p_tf: self.y_bounds_p,
                       self.t_bounds_p_tf: self.t_bounds_p,
                       self.p_bounds_tf: self.p_bounds,
                       self.t_body_tf: t_body_batch,
                       self.x_body_tf: x_body_batch,
                       self.y_body_tf: y_body_batch,
                       self.vn_body_tf: vn_body_batch,
                       self.nx_body_tf: nx_body_batch,
                       self.ny_body_tf: ny_body_batch,
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value[0][i],
                 loss_u[0][i],
                 loss_v[0][i],
                 loss_u_bounds[0][i],
                 loss_v_bounds[0][i],
                 loss_vn_body[0][i],
                 loss_p_bounds[0][i],
                 loss_e1[0][i],
                 loss_e2[0][i],
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.loss_u,
                                                       self.loss_v,
                                                       self.loss_u_bounds,
                                                       self.loss_v_bounds,
                                                       self.loss_vn_body,
                                                       self.loss_p_bounds,
                                                       self.loss_e1,
                                                       self.loss_e2,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value[0][i], elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
                i += 1

            it += 1

            if it >= imax:
                break

        scipy.io.savemat(save_path + 'Training_results_finer_res_%s.mat' % (
            time.strftime('%d_%m_%Y')),
                         {'Loss': loss_value, 'Loss_u': loss_u, 'Loss_v': loss_v, 'Loss_u_bounds': loss_u_bounds,
                          'Loss_v_bounds': loss_v_bounds, 'Loss_vn_body': loss_vn_body,
                          'Loss_p': loss_p_bounds,
                          'Loss_e1': loss_e1, 'Loss_e2': loss_e2, 'Time': running_time, 'Iter': it})

    def predict(self, t_star, x_star, y_star):
        
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star}
        
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return  u_star, v_star, p_star

    def predict_grad(self, t_star, x_star, y_star):

        tf_dict = {self.t_eqns_tf: t_star, self.x_eqns_tf: x_star, self.y_eqns_tf: y_star}

        u_x_star = self.sess.run(self.u_x_eqns_pred, tf_dict)
        u_y_star = self.sess.run(self.u_y_eqns_pred, tf_dict)
        v_x_star = self.sess.run(self.v_x_eqns_pred, tf_dict)
        v_y_star = self.sess.run(self.v_y_eqns_pred, tf_dict)

        return u_x_star, u_y_star, v_x_star, v_y_star

if __name__ == "__main__":

    ###############################################Input Data##########################################################

    #Specify path that contains the matfiles used for training
    data_path = '/scratch16/rni2/PINN/2D_Fish_Simulation/Revisions/Matfiles_Input/1) Network Size/'

    # Specify path for saving PINN outputs
    save_path = '/scratch16/rni2/PINN/2D_Fish_Simulation/Revisions/Results/'

    #Specify file name that contains x,y,t coordinates and velocity field
    vel_path = '2Dvel_data_cut.mat'

    #Specify file name that contains x,y,t coordinates and velocity field in select section
    vel_sec_path = '2Dvel_data_sec.mat'

    #Specify file name that contains x,y,t coordinates to regress NSE
    NSE_path = '2Dvel_data_full.mat'

    #Specify file name that contains surface points of body and corresponding normal velocity component
    body_path = 'body_norm_data.mat'

    #Specify file name that contains x,y,t points on boundary where a pressure boundary condition is enforced
    bounds_p_path = 'press_bound_data.mat'

    #Specify if an inlet velocity BC should be enforced. Set true for yes, false for no
    bounds_v_path = 'vel_bound_data.mat'

    #Enter the Reynolds Number
    Re  = 5000

    #Enter the training duration in hours (code will break when either total time or max iter is reached first)
    max_time  = 13

    #Enter the number of training iterations (code will break when either total time or max iter is reached first)
    max_it  = 232500
    #max_it  = 100

    batch_size = 10000
    #This is the mini-batch size for Adam's optimizer - determines how many data points are sampled at each iteration.
    #Lower if memory becomes an issue

    layers = [3] + 12*[120] + [3]
    # First layer has 3 nuerons because the inputs are x,y, t
    # Next 10 hidden layers have 50 neurons per output variable (decrease if memory is an issue)
    # Last layer 3 neurons since the outputs are u,v,p

    ###############################################Input Data##########################################################

    ################################################Load Data##########################################################

    #Load velocity field training data
    data = scipy.io.loadmat(data_path + vel_path) #loads matfile
    t_star = data['t_star'] # T x 1
    X_star = data['x_star'] # N x 1
    Y_star = data['y_star'] # N x 1
    T = t_star.shape[0] #returns number of rows of t
    N = X_star.shape[0] #returns number of rows of x
    # rows are the x-y location, columns are the progression in time
    U_star = data['U_star'] # N x T
    V_star = data['V_star'] # N x T
    T_star = np.tile(t_star, (1,N)).T # N x T #repeats t_star in a row N times, then take transpose

    data = scipy.io.loadmat(data_path + vel_sec_path)  # loads matfile
    t_sec = data['t_sec']  # T x 1
    X_sec = data['x_sec']  # N x 1
    Y_sec = data['y_sec']  # N x 1
    Tsec = t_sec.shape[0]  # returns number of rows of t
    Nsec = X_sec.shape[0]  # returns number of rows of x
    # rows are the x-y location, columns are the progression in time
    U_sec = data['U_sec']  # N x T
    V_sec = data['V_sec']  # N x T
    T_sec = np.tile(t_sec, (1, Nsec)).T  # N x T #repeats t_star in a row N times, then take transpose

    #Load body points and normal velocity data
    body = scipy.io.loadmat(data_path + body_path)  # loads matfile
    t_body = body['t_body']
    X_body = body['xmid']
    Y_body = body['ymid']
    Vn_body = body['vnorm']
    Nx_body = body['nx']
    Ny_body = body['ny']
    Tb = t_body.shape[0]  # returns number of rows of t
    Nb = X_body.shape[0]  # returns number of rows of x
    T_body = np.tile(t_body, (1, Nb)).T  # N x T #repeats t_star in a row N times, then take transpose

    # Load boundary coordinates and pressure
    b = np.zeros(2)
    if bounds_p_path:
        bounds_p = scipy.io.loadmat(data_path + bounds_p_path)  # loads matfile
        t_bounds_p = bounds_p['t_bounds']
        X_bounds_p = bounds_p['x_bounds_p']
        Y_bounds_p = bounds_p['y_bounds_p']
        P_bounds = bounds_p['p_bounds']
        N_bounds_p = X_bounds_p.shape[0]  # returns number of rows of x
        T_bounds_p = np.tile(t_bounds_p, (1, N_bounds_p)).T  # N x T #repeats t_star in a row N times, then take transpose
        b[1] = 1

    # Load boundary coordinates and velocity
    if bounds_v_path:
        bounds_v = scipy.io.loadmat(data_path + bounds_v_path)  # loads matfile
        t_bounds_v = bounds_v['t_bounds']
        X_bounds_v = bounds_v['x_bounds_v']
        Y_bounds_v = bounds_v['y_bounds_v']
        u_bounds = bounds_v['u_bounds']
        v_bounds = bounds_v['v_bounds']
        N_bounds_v = X_bounds_v.shape[0]  # returns number of rows of x
        T_bounds_v = np.tile(t_bounds_v, (1, N_bounds_v)).T  # N x T #repeats t_star in a row N times, then take transpose
        b[0] = 1

    ################################################Load Data##########################################################

    ##############################################Training Data########################################################

    #Velocity Data
    T_data = T
    N_data = N
    #randomly mix the velocity field data
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    u_data = U_star[:, idx_t][idx_x,:].flatten()[:,None]
    v_data = V_star[:, idx_t][idx_x,:].flatten()[:,None]

    ind = np.where(np.isnan(u_data))[0] #remove points that are within the body
    t_data = np.delete(t_data, ind[:], 0)
    x_data = np.delete(x_data, ind[:], 0)
    y_data = np.delete(y_data, ind[:], 0)
    u_data = np.delete(u_data, ind[:], 0)
    v_data = np.delete(v_data, ind[:], 0)

    # Velocity Data
    T_data_sec = Tsec
    N_data_sec = Nsec
    # randomly mix the velocity field data
    idx_t = np.concatenate([np.array([0]), np.random.choice(T - 2, T_data_sec - 2, replace=False) + 1, np.array([Tsec - 1])])
    idx_x = np.random.choice(Nsec, N_data_sec, replace=False)
    t_sec = T_sec[:, idx_t][idx_x, :].flatten()[:, None]
    x_sec = X_sec[:, idx_t][idx_x, :].flatten()[:, None]
    y_sec = Y_sec[:, idx_t][idx_x, :].flatten()[:, None]
    u_sec = U_sec[:, idx_t][idx_x, :].flatten()[:, None]
    v_sec = V_sec[:, idx_t][idx_x, :].flatten()[:, None]

    ind = np.where(np.isnan(u_sec))[0]  # remove points that are within the body
    t_sec = np.delete(t_sec, ind[:], 0)
    x_sec = np.delete(x_sec, ind[:], 0)
    y_sec = np.delete(y_sec, ind[:], 0)
    u_sec = np.delete(u_sec, ind[:], 0)
    v_sec = np.delete(v_sec, ind[:], 0)

    #Body Data
    t_body = T_body.flatten()[:, None]  # NT x 1
    x_body = X_body.flatten()[:, None]  # NT x 1
    y_body = Y_body.flatten()[:, None]  # NT x 1
    vn_body = Vn_body.flatten()[:, None]  # NT x 1
    nx_body = Nx_body.flatten()[:, None]  # NT x 1
    ny_body = Ny_body.flatten()[:, None]  # NT x 1

    #External Boundary Data (Pressure)
    if bounds_p_path:
        t_bounds_p = T_bounds_p.flatten()[:,None] # NT x 1
        x_bounds_p = X_bounds_p.flatten()[:,None] # NT x 1
        y_bounds_p = Y_bounds_p.flatten()[:,None] # NT x 1
        p_bounds = P_bounds.flatten()[:,None] # NT x 1
    else:
        t_bounds_p = np.zeros((2,1))
        x_bounds_p = np.zeros((2,1))
        y_bounds_p = np.zeros((2,1))
        p_bounds = np.zeros((2,1))

    #External Boundary Data (Velocity
    if bounds_v_path:
        t_bounds_v = T_bounds_v.flatten()[:,None] # NT x 1
        x_bounds_v = X_bounds_v.flatten()[:,None] # NT x 1
        y_bounds_v = Y_bounds_v.flatten()[:,None] # NT x 1
        u_bounds = u_bounds.flatten()[:,None] # NT x 1
        v_bounds = v_bounds.flatten()[:,None] # NT x 1
    else:
        t_bounds_v = np.zeros((2,1))
        x_bounds_v = np.zeros((2,1))
        y_bounds_v = np.zeros((2,1))
        u_bounds = np.zeros((2,1))
        v_bounds = np.zeros((2,1))

    #NSE

    data = scipy.io.loadmat(data_path + NSE_path) #loads matfile
    t_star = data['t_star']  # T x 1
    X_star = data['x_star']  # N x 1
    Y_star = data['y_star']  # N x 1
    T = t_star.shape[0]  # returns number of rows of t
    N = X_star.shape[0]  # returns number of rows of x
    U_star = data['U_star']  # N x T
    V_star = data['V_star']  # N x T
    T_star = np.tile(t_star, (1, N)).T  # N x T #repeats t_star in a row N times, then take transpose

    T_eqns = T
    N_eqns = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    u_eqns = U_star[:, idx_t][idx_x,:].flatten()[:,None]

    ind2 = np.where(np.isnan(u_eqns))[0]
    t_eqns = np.delete(t_eqns, ind2[:], 0)
    x_eqns = np.delete(x_eqns, ind2[:], 0)
    y_eqns = np.delete(y_eqns, ind2[:], 0)

    ##############################################Training Data########################################################

    #################################################Training##########################################################
    model = HFM(t_data, x_data, y_data, u_data, v_data,
                t_sec, x_sec, y_sec, u_sec, v_sec,
                t_eqns, x_eqns, y_eqns,
                t_bounds_v, x_bounds_v, y_bounds_v, u_bounds, v_bounds,
                t_bounds_p, x_bounds_p, y_bounds_p, p_bounds,
                t_body, x_body, y_body, vn_body, nx_body, ny_body,
                layers, batch_size, save_path,
                Rey = Re)
    
    model.train(imax = max_it, total_time = max_time, learning_rate=1e-3)
    #################################################Training##########################################################

    ################################################Save Data##########################################################

    data = scipy.io.loadmat(data_path +  NSE_path)  # loads matfile
    t_star = data['t_star']  # T x 1
    X_star = data['x_star']  # N x 1
    Y_star = data['y_star']  # N x 1
    T = t_star.shape[0]  # returns number of rows of t
    N = X_star.shape[0]  # returns number of rows of x
    U_star = data['U_star']  # N x T
    V_star = data['V_star']  # N x T
    T_star = np.tile(t_star, (1, N)).T  # N x T #repeats t_star in a row N times, then take transpose

    error_u = np.zeros([1, t_star.shape[0]])
    error_v = np.zeros([1, t_star.shape[0]])

   #Creates variables of the same size but all zeros

    t = T_star.flatten()[:, None]  # NT x 1
    x = X_star.flatten()[:, None]  # NT x 1
    y = Y_star.flatten()[:, None]  # NT x 1
    u = U_star.flatten()[:, None]  # NT x 1
    v = V_star.flatten()[:, None]  # NT x 1

    ind3 = np.where(np.isnan(u))[0]
    t = np.delete(t, ind3[:], 0)
    x = np.delete(x, ind3[:], 0)
    y = np.delete(y, ind3[:], 0)
    u = np.delete(u, ind3[:], 0)
    v = np.delete(v, ind3[:], 0)

    #Compute error in velocity field predictions
    for snap in range(0,t_star.shape[0]):
        t_ind = np.where(t == t_star[snap])[0]

        t_test = t[t_ind[:]]
        x_test = x[t_ind[:]]
        y_test = y[t_ind[:]]

        u_test = u[t_ind[:]]
        v_test = v[t_ind[:]]

        # Prediction
        u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
    
        # Error
        error_u[0][snap] = relative_error(u_pred, u_test)
        error_v[0][snap] = relative_error(v_pred, v_test)

    #Compute predictions in format suitable for saving
    U_pred = np.zeros(T_star.shape)
    V_pred = np.zeros(T_star.shape)
    P_pred = np.zeros(T_star.shape)
    Ux_pred = np.zeros(T_star.shape)
    Uy_pred = np.zeros(T_star.shape)
    Vx_pred = np.zeros(T_star.shape)
    Vy_pred = np.zeros(T_star.shape)

    for snap in range(0, t_star.shape[0]):
        t_test = T_star[:, snap:snap + 1]
        x_test = X_star[:, snap:snap + 1]
        y_test = Y_star[:, snap:snap + 1]

        # Prediction
        u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
        ux_pred, uy_pred, vx_pred, vy_pred = model.predict_grad(t_test, x_test, y_test)

        # Storing results for each  time step FOR MATLAB OUTPUT

        ind = np.where(np.isnan(U_star[:, snap:snap + 1]))[0]
        u_pred[ind] = np.nan
        v_pred[ind] = np.nan
        p_pred[ind] = np.nan

        ind = np.where(np.isnan(U_star[:, snap:snap + 1]))[0]
        ux_pred[ind] = np.nan
        uy_pred[ind] = np.nan
        vx_pred[ind] = np.nan
        vy_pred[ind] = np.nan

        U_pred[:, snap:snap + 1] = u_pred
        V_pred[:, snap:snap + 1] = v_pred
        P_pred[:, snap:snap + 1] = p_pred

        Ux_pred[:, snap:snap + 1] = ux_pred
        Uy_pred[:, snap:snap + 1] = uy_pred
        Vx_pred[:, snap:snap + 1] = vx_pred
        Vy_pred[:, snap:snap + 1] = vy_pred

    #Body Predictions
    body = scipy.io.loadmat(data_path + 'body_data.mat')  # loads matfile
    t_body = body['t_body']
    X_body = body['x_body']
    Y_body = body['y_body']
    Tb = t_body.shape[0]  # returns number of rows of t
    Nb = X_body.shape[0]  # returns number of rows of x
    T_body = np.tile(t_body, (1, Nb)).T  # N x T #repeats t_star in a row N times, then take transpose

    P_pred_body = np.zeros(T_body.shape)

    for snap in range(0, Tb):
        t_test = T_body[:,snap:snap+1]
        x_test = X_body[:,snap:snap+1]
        y_test = Y_body[:,snap:snap+1]

        u_pred_body, v_pred_body, p_pred_body = model.predict(t_test, x_test, y_test)

        P_pred_body[:, snap:snap+1] = p_pred_body

    scipy.io.savemat(save_path + 'PINN_2D_UndulatingBody_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'U_pred': U_pred, 'V_pred':V_pred, 'P_pred':P_pred,
                      'Ux_pred': Ux_pred, 'Uy_pred': Uy_pred, 'Vx_pred': Vx_pred, 'Vy_pred': Vy_pred,
                      'P_pred_body': P_pred_body,
                      'Err_u': error_u, 'Err_v': error_v})

    ################################################Save Data##########################################################
