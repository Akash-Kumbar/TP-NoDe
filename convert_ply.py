import open3d as o3d
import numpy as np

xyz= np.loadtxt('')
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(xyz)

o3d.io.write_point_cloud(".ply",pcd)
