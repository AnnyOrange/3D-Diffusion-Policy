import visualizer
import os
import zarr
root_dir = "../3D-Diffusion-Policy/data/"
env_name = "basketball"
save_dir=os.path.join(root_dir, 'metaworld_'+env_name+'_expert.zarr')
zarr_root_dir = save_dir
print(save_dir)
zarr_data_dir = zarr_root_dir + '/data'
zarr_meta_dir = zarr_root_dir + '/meta'
zarr_data = zarr.open(zarr_data_dir, mode='r')

your_pointcloud = zarr_data['point_cloud'][:] # your point cloud data, numpy array with shape (N, 3) or (N, 6)
print(your_pointcloud.shape)
visualizer.visualize_pointcloud(your_pointcloud)