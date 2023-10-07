import torch
import numpy as np
from torch_cluster import fps
from pytorch3d.ops import knn_points, ball_query
import open3d as o3d
import polyscope as ps
ps.init()

def visPC(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])

### Normalize point cloud
def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance

def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    indices = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]
        sampled.append(pcls[i:i+1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    return sampled, indices

def Minkowski_distance(src, dst, p):
    """
    Calculate Minkowski distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point Minkowski distance, [B, N, M]
    """
    return torch.cdist(src,dst,p=p)

def gather_idx(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def gather_idx(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def get_dist(src, dst):
    """
    Calculate the Euclidean distance between each point pair in two point clouds.
    Inputs:
        src[B, M, 3]: point cloud 1
        dst[B, N, 3]: point cloud 2
    Return: 
        dist[B, M, N]: distance matrix
    """
    print(src.shape)
    exit()
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def dilated_ball_queryOG(dist, h, base_radius, max_radius):
    '''
    Density-dilated ball query
    Inputs:
        dist[B, M, N]: distance matrix 
        h(float): bandwidth
        base_radius(float): minimum search radius
        max_radius(float): maximum search radius
    Returns:
        radius[B, M, 1]: search radius of point
    '''

    # kernel density estimation (Eq. 8)
    sigma = 1
    gauss = torch.exp(-(dist)/(2*(h**2)*(sigma**2))) # K(x-x_i/h), [B, M, N]
    kd_dist = torch.sum(gauss, dim=-1).unsqueeze(-1) # kernel distance, [B, M, 1]

    # normalization
    kd_score = kd_dist / (torch.max(kd_dist, dim=1)[0].unsqueeze(-1) + 1e-9) # [B, M, 1]
    radius = base_radius + (max_radius - base_radius)*kd_score # kd_score -> max, base_radius -> max_radius

    return radius

def dilated_ball_query(dist, h, base_radius, max_radius):
    '''
    Density-dilated ball query
    Inputs:
        dist[B, M, N]: distance matrix 
        h(float): bandwidth
        base_radius(float): minimum search radius
        max_radius(float): maximum search radius
    Returns:
        radius[B, M, 1]: search radius of point
    '''

    # kernel density estimation (Eq. 8)
    gauss = 0.5 + ((0.5 * torch.sgn(dist)) * (1 - torch.exp(-dist/h))) 
    
    # gauss = torch.exp(-(dist)/(2*(h**2)*(sigma**2))) # K(x-x_i/h), [B, M, N]
    kd_dist = torch.sum(gauss, dim=-1).unsqueeze(-1) # kernel distance, [B, M, 1]

    # normalization
    kd_score = kd_dist / (torch.max(kd_dist, dim=1)[0].unsqueeze(-1) + 1e-9) # [B, M, 1]
    radius = base_radius + (max_radius - base_radius)*kd_score # kd_score -> max, base_radius -> max_radius

    return radius

def density_aware_knn(x, min_k, max_k, seed_k=3):
    base_index = min_k
    max_index = max_k
    N = x.shape[1]
    approx_patch_size = int((min_k + max_k) // 2)
    ncentroids = int(seed_k*N/approx_patch_size)
    centroid,_ = farthest_point_sampling(x, ncentroids)
    dist = get_dist(centroid, x)
    sigma = 1
    h=0.1
    gauss = torch.exp(-(dist)/(2*(h**2)*(sigma**2))) # K(x-x_i/h), [B, M, N]
    kd_dist = torch.sum(gauss, dim=-1).unsqueeze(-1) # kernel distance, [B, M, 1]
    kd_score = kd_dist / (torch.max(kd_dist, dim=1)[0].unsqueeze(-1) + 1e-9) # [B, M, 1]
    ks = torch.ceil(base_index + (max_index - base_index) * kd_score).to(torch.int).squeeze(0).squeeze(1)
    patches = []
    indices = []
    for i in range(ks.shape[0]):
        _, idx, points = knn_points(centroid, x, K=ks[i].item(), return_nn=True)
        
        idx = idx.squeeze(0)
        points = points.squeeze(0)
        patches.append(points[i])
        indices.append(idx[i])
        # patches.append(knn_points(centroid, x, K=ks[i].item(), return_nn=True)[2].squeeze(0)[i])
    
    
    return patches, indices


### Convert point cloud to patches
def convertToPatchKNN(xyz, patch_size=256, seed_k=3):
    N, d = xyz.size()
    xyz = xyz.unsqueeze(0)
    seed_pnts, _ = farthest_point_sampling(xyz, int(seed_k * N / patch_size))
    _, _, patches = knn_points(seed_pnts, xyz, K=patch_size, return_nn=True)
    patches = patches[0]
    return patches

def convertToPatchBQ(xyz, radius=0.2, max_patch_size=128, seed_k=4):
    N, d = xyz.size()
    xyz = xyz.unsqueeze(0)
    ncentroids = int(seed_k*N/max_patch_size)
    seed_pnts, _ = farthest_point_sampling(xyz, ncentroids)
    _,_,patches = ball_query(seed_pnts, xyz, K=max_patch_size,radius=radius, return_nn=True)
    patches = patches[0]
    return patches

def convertToPatchDilatedBQ(xyz, base_radius=0.05, seed_k=3):
    N, d = xyz.size()
    approx_patch_size = 256
    xyz = xyz.unsqueeze(0)
    ncentroids = int(seed_k*N/approx_patch_size)
    centroid,_ = farthest_point_sampling(xyz, ncentroids)
    dist = get_dist(centroid, xyz)
    radius = dilated_ball_query(dist, h=0.1, base_radius=base_radius, max_radius=base_radius*3)
    mask = (dist < radius).float().squeeze(0)
    xyz = xyz.squeeze(0)
    patches = []
    for i in range(ncentroids):
        indices = (mask[i] == 1).nonzero(as_tuple=True)[0]  
        patch = xyz[indices]
        patches.append(patch)
    return patches

def convertToPatchDAKNN(xyz, min_k = 64, max_k = 512, seed_k = 3):
    xyz = xyz.unsqueeze(0)
    return density_aware_knn(xyz, min_k, max_k)

def readOff(path, n):
    meshD = o3d.io.read_triangle_mesh(path) 
    pcd = meshD.sample_points_uniformly(n)
    xyz = np.array(pcd.points, dtype=np.float32)

    return xyz

def polyRenderPC(xyz, colors, radius = 0.005):
    ps_cloud = ps.register_point_cloud("my points", xyz, enabled=True, radius=radius)
    # basic color visualization
    ps_cloud.add_color_quantity("color_meant", colors)
    ps.show() 

if __name__ == '__main__':
    temp = torch.rand((1, 10000, 3)).cuda()
    # print(convertToPatchMKNN(temp, 256).shape)
    sit = density_aware_knn(temp, 64, 512)
    for i in sit:
        print(i.shape)
    