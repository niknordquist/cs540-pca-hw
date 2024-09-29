from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    
    x = np.load(filename) # Load data from filename into numpy array
    mu = np.mean(x, axis=0) # Calculate μx
    return x - mu
    
def get_covariance(dataset):
    # Your implementation goes here!
    return np.dot(np.transpose(dataset), dataset)

def get_eig(S, m):
    # Your implementation goes here!
    
    raise NotImplementedError

def get_eig_prop(S, prop):
    # Your implementation goes here!
    raise NotImplementedError

def project_image(image, U):
    # Your implementation goes here!
    raise NotImplementedError

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, ax1, ax2 = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    raise NotImplementedError

def perturb_image(image, U, sigma):
    # Your implementation goes here!
    raise NotImplementedError

def combine_image(image1, image2, U, lam):
    # Your implementation goes here!
    raise NotImplementedError

x = load_and_center_dataset("face_dataset.npy")
S = (get_covariance(x))
Lambda, U = get_eig(S, 2)
