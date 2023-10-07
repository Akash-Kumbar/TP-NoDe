import numpy as np
import torch
import pytorch3d.ops as ops
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, KDTree

def genGaussianNoise(xyz, std=0.015):
    noise = torch.randn_like(xyz) * std
    return xyz + noise

def genLaplacianNoise(xyz, std=0.01):
    noise = torch.from_numpy(np.random.laplace(0, std, size=xyz.shape)).cuda()
    # print(type(noise))
    return xyz + noise

def genDiscreteNoise(xyz, std=0.01):
    scale = std
    prob = 0.1
    template = np.array([
            [scale, 0, 0],
            [-scale, 0, 0],
            [0, scale, 0],
            [0, -scale, 0],
            [0, 0, scale],
            [0, 0, -scale],
        ], dtype=np.float32)
    num_points = xyz.shape[0]
    uni_rand = np.random.uniform(size=num_points)
    noise = np.zeros([num_points, 3])
    for i in range(template.shape[0]):
            idx = np.logical_and(0.1*i <= uni_rand, uni_rand < 0.1*(i+1))
            noise[idx] = template[i].reshape(1, 3)
    noise = torch.FloatTensor(noise).to(xyz)
    return xyz + noise

def genUniformBallNoise(xyz, std):
    scale = std
    N = xyz.shape[0]
    phi = np.random.uniform(0, 2*np.pi, size=N)
    costheta = np.random.uniform(-1, 1, size=N)
    u = np.random.uniform(0, 1, size=N)
    theta = np.arccos(costheta)
    r = scale * u ** (1/3)
    noise = np.zeros([N, 3])
    noise[:, 0] = r * np.sin(theta) * np.cos(phi)
    noise[:, 1] = r * np.sin(theta) * np.sin(phi)
    noise[:, 2] = r * np.cos(theta)
    noise = torch.FloatTensor(noise).to(xyz)
    return xyz + noise

def genCovNoise(xyz, std):
    num_points = xyz.shape[0]
    cov = np.cov(xyz.cpu().numpy().T)
    cov = torch.FloatTensor(cov)
    # print(cov.shape)
    
    # exit()
    std_factor = std
    noise = np.random.multivariate_normal(np.zeros(3), cov.numpy(), num_points)
    noise = torch.FloatTensor(noise).to(xyz)
    return xyz + noise*std_factor


def addNoiseToPC(xyz, std, noise_type='Gaussian'):
    if noise_type == 'Gaussian':
         noise = genGaussianNoise(xyz, std)
    elif noise_type == 'Laplacian':
         noise = genLaplacianNoise(xyz, std)
    elif noise_type == 'Discrete':
        noise = genDiscreteNoise(xyz, std)
    elif noise_type == 'UniformBall':
        noise = genUniformBallNoise(xyz, std)
    elif noise_type == 'Covariance':
        noise = genCovNoise(xyz, std) 
    return torch.cat([xyz, noise], dim=0)