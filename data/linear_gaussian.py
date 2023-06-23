from torch.utils.data import Dataset
import numpy as np
from scipy.stats import norm

# Higher dimensional linear gaussian data, randomized parameters
class LinearGaussian(Dataset):
    
    # size: number of datapoints per param
    # dim: number of param configurations
    def __init__(self, size, dim, suppress_print=False, param_seed=0, generation_seed=0):

        np.random.seed(param_seed)
        
        self.size = size
        self.dim = dim
        
        self.c_zx = np.random.uniform(0, 5, dim)
        self.c_zy = np.random.uniform(-10, -5, dim)
        self.c_xy = np.random.uniform(0, 3, dim)
        self.sig_x = np.random.uniform(0, 1, dim)
        self.sig_y = np.random.uniform(0, 3, dim)
        
        if not suppress_print:
            print('Linear Gaussian Parameters')
            print(f'c_zx:{self.c_zx}\nc_zy:{self.c_zy}\nc_xy:{self.c_xy}\nsig_x:{self.sig_x}\nsig_y:{self.sig_y}\n\n')

        self.generate_linear_gaussian_backdoor(size, dim, generation_seed)


    def generate_linear_gaussian_backdoor(self, size, dim, seed, do_x=None, do_y=None, do_z=None):

        np.random.seed(seed)
        if do_z is None:
            Z = np.random.normal(0, 1, (size, dim))
        else:
            Z = do_z
        
        if do_x is None:
            X = np.random.normal(self.c_zx * Z, self.sig_x)
        else:
            X = do_x
        
        assert X.shape == Z.shape
            
        if do_y is None:
            Y = np.random.normal(self.c_zy * Z + self.c_xy * X, self.sig_y)
        else:
            Y = do_y
        
        assert Y.shape == Z.shape
        
        self.Z = np.array(Z, dtype='float32')
        self.X = np.array(X, dtype='float32')
        self.Y = np.array(Y, dtype='float32')
        
    def ground_truth_cond_likelihood(self, y, x):
        return norm(loc=(self.c_zy * self.c_zx / (self.c_zx ** 2
                    + self.sig_x ** 2) + self.c_xy) * x,
                    scale=np.sqrt(self.c_zy ** 2 * self.sig_x ** 2 / (self.c_zx ** 2 
                    + self.sig_x ** 2) + self.sig_y ** 2)).pdf(y)
    
    def ground_truth_do_likelihood(self, y, do_x):
        return norm(loc=self.c_xy * do_x, scale=np.sqrt(
                self.sig_y ** 2 + self.c_zy ** 2)).pdf(y)
    
    def ground_truth_joint_y_x(self, y, x):
        return self.ground_truth_cond_likelihood(y, x) * self.ground_truth_x_marg(x)
            
    
    def ground_truth_joint_x_y_z(self, x, y, z):
        return norm().pdf(z) * norm(loc=self.c_zx * z, scale=self.sig_x).pdf(x) * norm(
            loc=self.c_zy * z + self.c_xy * x, scale=self.sig_y).pdf(y)
    
    def ground_truth_x_marg(self, x):
        return norm(loc=0, scale=np.sqrt(self.c_zx ** 2 + self.sig_x ** 2)).pdf(x)
    
    def ground_truth_y_marg(self, y):
        return norm(loc=0, scale=np.sqrt(
            self.c_xy ** 2 * self.c_zx ** 2 + self.c_zy ** 2 
            + self.sig_y ** 2 + self.c_xy ** 2 * self.sig_x ** 2)).pdf(y)
    
    def ground_truth_y_given_x_z(self, y, x, z):
        return norm(loc=self.c_zy * z + self.c_xy * x, scale=self.sig_y).pdf(y)
    
    def ground_truth_z_given_x_y(self, z, x, y):
        return self.ground_truth_joint_x_y_z(x, y, z) / self.ground_truth_joint_y_x(y, x)

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, n):
        return self.X[n], self.Y[n], self.Z[n]