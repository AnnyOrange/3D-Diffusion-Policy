# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
import wandb.sdk.data_types.video as wv
import matplotlib.pyplot as plt
from termcolor import cprint
import copy
import imageio
from metaworld.policies import *
import wandb
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

# import faulthandler
# faulthandler.enable()

# seed = np.random.randint(0, 100)
# seeds = 73

# print(seed)
def load_mw_policy(task_name):
	if task_name == 'peg-insert-side':
		agent = SawyerPegInsertionSideV2Policy()
	else:
		task_name = task_name.split('-')
		task_name = [s.capitalize() for s in task_name]
		task_name = "Sawyer" + "".join(task_name) + "V2Policy"
		agent = eval(task_name)()
	return agent



def save_video_to_file(videos, save_path, fps=10):
    if len(videos.shape) == 5:
        videos = videos[:, 0]  # 如果 shape 是 5 维，去掉 batch 维度
    # print(videos.shape)
    videos = np.transpose(videos, (0, 2, 3, 1)) 
    with imageio.get_writer(save_path, fps=fps) as video_writer:
        for frame in videos:
            video_writer.append_data(frame)
    print(f"Video saved at {save_path}")



def qpos_apos(state_arrays_sub,action_arrays_sub,save_dir,episode_idx):
    save_path = os.path.join(save_dir,'qpos_apos',f'{episode_idx}_qpos_apos.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    qpos = np.array(action_arrays_sub)
    apos = np.array(state_arrays_sub)
    
    # 创建3D图表
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制 qpos 曲线
    ax.plot(qpos[:, 0], qpos[:, 1], qpos[:, 2], label='qpos', color='blue', marker='o')
    
    # 绘制 apos 曲线
    ax.plot(apos[:, 0], apos[:, 1], apos[:, 2], label='apos', color='red', marker='x')
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Qpos and Apos 3D Trajectories')
    ax.legend(loc='best')
    ax.grid(True)
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    # print(f"Plot saved at: {save_path}")
    
def main(args):
	save_video=True
	env_name = args.env_name

	save_dir = os.path.join(args.root_dir, 'metaworld_'+args.env_name+'_expert.zarr')
	# save_dir_2 = os.path.join(save_dir,'seed.txt')
	# print(f"Saving seed to: {save_dir_2}")
	# with open(save_dir_2, 'w') as f:
	# 	f.write(str(seed))
	if os.path.exists(save_dir):
		cprint('Data already exists at {}'.format(save_dir), 'red')
		cprint("If you want to overwrite, delete the existing directory first.", "red")
		cprint("Do you want to overwrite? (y/n)", "red")
		user_input = 'y'
		if user_input == 'y':
			cprint('Overwriting {}'.format(save_dir), 'red')
			os.system('rm -rf {}'.format(save_dir))
		else:
			cprint('Exiting', 'red')
			return
	os.makedirs(save_dir, exist_ok=True)
	# video_save_dir = os.path.join(save_dir, 'videos')  # 视频保存路径
    # os.makedirs(video_save_dir, exist_ok=True)
	# print("main",env_name)
	e = SimpleVideoRecordingWrapper(MetaWorldEnv(task_name=env_name, device="cuda:0", use_point_crop=True, seed = 73))
	
	num_episodes = args.num_episodes
	cprint(f"Number of episodes : {num_episodes}", "yellow")
	
	
	total_count = 0
	img_arrays = []
	point_cloud_arrays = []
	depth_arrays = []
	state_arrays = []
	full_state_arrays = []
	action_arrays = []
	episode_ends_arrays = []
	info_arrays = []
	info_grasp_arrays = []
	info_near_arrays = []
	total_subs = []
	episode_idx = 0
	all_success_rates = []
	all_traj_rewards = []
	qpos_arrays = []
	save_qpos_arrays = []
	apos_arrays = []
	apos_gripper_arrays = []
	qpos_gripper_arrays = []

	mw_policy = load_mw_policy(env_name)
	raw_state_file = os.path.join(save_dir, 'raw_state.txt')
	# print(mw_policy)
	
	# loop over episodes
	with open(raw_state_file, 'a') as raw_state_log:
		while episode_idx < num_episodes:
			# seed = episode_idx
			raw_state = e.reset()['full_state']
			raw_state_log.write(f'{episode_idx}: raw_state = {raw_state.tolist()}\n')
			obs_dict = e.get_visual_obs()

			
			done = False
			
			ep_reward = 0.
			ep_success = False
			ep_success_times = 0
			

			img_arrays_sub = []
			point_cloud_arrays_sub = []
			depth_arrays_sub = []
			state_arrays_sub = []
			full_state_arrays_sub = []
			action_arrays_sub = []
			info_arrays_sub = []
			info_near_arrays_sub = []
			info_grasp_arrays_sub = []
			apos_sub = []
			qpos_sub = []
			apos_gripper_sub = []
			qpos_gripper_sub = []
			total_count_sub = 0
	
			while not done:

				total_count_sub += 1
				
				obs_img = obs_dict['image']
				obs_robot_state = obs_dict['agent_pos']
				# print(obs_robot_state)
				obs_point_cloud = obs_dict['point_cloud']
				obs_depth = obs_dict['depth']
				# obs_robot_state_sub.append(obs_robot_state)

				img_arrays_sub.append(obs_img)
				point_cloud_arrays_sub.append(obs_point_cloud)
				depth_arrays_sub.append(obs_depth)
				state_arrays_sub.append(obs_robot_state)
				full_state_arrays_sub.append(raw_state)
				
				action = mw_policy.get_action(raw_state)
				qpos = action[:3]/100+raw_state[:3]
				# qpos = action[:3]+raw_state[:3]
				# print("raw_state",raw_state[:3])
				apos_sub.append(raw_state[:3])
				qpos_sub.append(qpos)
				if action[3]==-1:
					qpos_gripper = 1
				else:
					qpos_gripper = 0.6
				apos_gripper_sub.append(raw_state[3])
				qpos_gripper_sub.append(qpos_gripper)
				# print(raw_state[:3]-qpos)
				# print("action",action[:3]+raw_state[:3])
				action_arrays_sub.append(action)
				obs_dict, reward, done, info = e.step(action)
				# current_mocap_pos = e.data.get_mocap_pos('mocap')
				# print("current_mocap_pos",current_mocap_pos)
				# print("obs",len(obs_dict['full_state']))
				# print("action",action)
				# print("action",action)
				# print("obs_dict",obs_dict)
				# print("info",info)
				# print("info",info['grasp_success'])
				info_near_arrays_sub.append(info['near_object'])
				info_grasp_arrays_sub.append(info['grasp_success'])

				raw_state = obs_dict['full_state']
				ep_reward += reward
	
				ep_success = ep_success or info['success']
				ep_success_times += info['success']
	
				if done:
					# print("total_sub",total_count_sub)
					# raw_state_log.write(f'{episode_idx}: last_action = {action.tolist()}\n')
					total_subs.append(total_count_sub)
					# print(info_near_arrays_sub)
					break
			all_success_rates.append(ep_success)
			all_traj_rewards.append(ep_reward)
			qpos_arrays.append(qpos_sub)
			apos_arrays.append(apos_sub)
			apos_gripper_arrays.append(apos_gripper_sub)
			qpos_gripper_arrays.append(qpos_gripper_sub)
			save_qpos_arrays.extend(copy.deepcopy(qpos_sub))
			# print(info['target_pos'])
			if save_video:
				videos = e.get_video()  # 获取视频帧
				if len(videos.shape) == 5:
					videos = videos[:, 0]  # 去掉 batch 维度
				
				video_path = os.path.join(save_dir, 'videos', f'episode_{episode_idx}.mp4')  # 保存路径
				os.makedirs(os.path.dirname(video_path), exist_ok=True)  # 确保目录存在
				save_video_to_file(videos, video_path, fps=10)  # 保存视频
			# print("len(action)",len(action_arrays_sub))
			if not ep_success or ep_success_times < 5:
				cprint(f'Episode: {episode_idx} failed with reward {ep_reward} and success times {ep_success_times}', 'red')
				qpos_apos(apos_sub,qpos_sub,save_dir,episode_idx)
				total_count += total_count_sub
				episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
				img_arrays.extend(copy.deepcopy(img_arrays_sub))
				point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
				depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
				state_arrays.extend(copy.deepcopy(state_arrays_sub))
				action_arrays.extend(copy.deepcopy(action_arrays_sub))
				info_near_arrays.extend(copy.deepcopy(info_near_arrays_sub))
				info_grasp_arrays.extend(copy.deepcopy(info_grasp_arrays_sub))
				full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))
				
				# continue
				# if save_video:
				# 	videos = e.get_video()
				# 	if len(videos.shape) == 5:
				# 		videos = videos[:, 0]  # 去掉 batch 维度

				# 	video_path = os.path.join(video_save_dir, f'episode_{episode_idx}.mp4')
				# 	save_video_to_file(videos, video_path) 
				episode_idx += 1
    
			else:
				qpos_apos(apos_sub,qpos_sub,save_dir,episode_idx)
				total_count += total_count_sub
				episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
				img_arrays.extend(copy.deepcopy(img_arrays_sub))
				point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
				depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
				state_arrays.extend(copy.deepcopy(state_arrays_sub))
				action_arrays.extend(copy.deepcopy(action_arrays_sub))
				info_near_arrays.extend(copy.deepcopy(info_near_arrays_sub))
				info_grasp_arrays.extend(copy.deepcopy(info_grasp_arrays_sub))
				full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))
				
				# print("ngroup",n_groups)
				
				cprint('Episode: {}, Reward: {}, Success Times: {}'.format(episode_idx, ep_reward, ep_success_times), 'green')
				save_dir_3 = os.path.join(save_dir,'succ_reward.txt')
				with open(save_dir_3, 'a') as f:
					f.write('Episode: {}, Reward: {}, Success Times: {}\n'.format(episode_idx, ep_reward, ep_success_times))
				episode_idx += 1
	
	log_data = dict()	
	log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
	log_data['mean_success_rates'] = np.mean(all_success_rates)	
	# videos = e.get_video()
	# if len(videos.shape) == 5:
	# 	videos = videos[:, 0] 
	
	# if save_video:
	# 	videos_wandb = wandb.Video(videos, fps=10, format="mp4")
	# 	log_data[f'sim_video_eval'] = videos_wandb
	# videos = None
	log_file_path = os.path.join(save_dir, 'logdata.txt')
	with open(log_file_path, 'w') as f:
		for key, value in log_data.items():
			f.write(f"{key}: {value}\n")
 
	# todo:save一下info['target_pos']然后save到os.path.join(save_dir,target_pos)
	save_path = os.path.join(save_dir, 'target_pos.npy')
	np.save(save_path, info['target_pos'])
	save_path = os.path.join(save_dir,'sim_pos.npy')
	print(len(save_qpos_arrays))
	np.save(save_path, save_qpos_arrays)
	n_groups = qpos.shape[-1]
	print(len(qpos_arrays))
	start = 0
	for idx in range(num_episodes):
		target_pos_sub = info['target_pos'][start:start+total_subs[idx]]
		qpos_plot(save_dir,n_groups,qpos_arrays[idx],apos_arrays[idx],apos_gripper_arrays[idx],qpos_gripper_arrays[idx],target_pos_sub,idx)
		start = start+total_subs[idx]
	# save data
 	###############################
    # save data
    ###############################
    # create zarr file
	zarr_root = zarr.group(save_dir)
	zarr_data = zarr_root.create_group('data')
	zarr_meta = zarr_root.create_group('meta')
	# save img, state, action arrays into data, and episode ends arrays into meta
	img_arrays = np.stack(img_arrays, axis=0)
	if img_arrays.shape[1] == 3: # make channel last
		img_arrays = np.transpose(img_arrays, (0,2,3,1))
	state_arrays = np.stack(state_arrays, axis=0)
	full_state_arrays = np.stack(full_state_arrays, axis=0)
	point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
	depth_arrays = np.stack(depth_arrays, axis=0)
	action_arrays = np.array(action_arrays)
	# print("actions_array.shape",action_arrays.shape)
	action_arrays = np.stack(action_arrays, axis=0)
	# info_arrays = np.array(info_arrays)
	# info_arrays = np.stack(info_arrays, axis=0)
	# if info_arrays.ndim == 1:
	# 	info_arrays = info_arrays[:, np.newaxis]
	episode_ends_arrays = np.array(episode_ends_arrays)

	compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
	img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
	state_chunk_size = (100, state_arrays.shape[1])
	full_state_chunk_size = (100, full_state_arrays.shape[1])
	point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
	depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
	action_chunk_size = (100, action_arrays.shape[1])
	# info_chunk_size = (100, info_arrays.shape[1])  # 根据info的字段数量设置chunk大小
	total_sub_chunk_size = (200,)  # 设置chunk大小，通常可以设置为(100,)或其他合适的值
	zarr_data.create_dataset('total_sub', data=total_subs, chunks=total_sub_chunk_size, dtype='int64', overwrite=True)
	zarr_data.create_dataset('info_near', data=info_near_arrays, chunks=total_sub_chunk_size, dtype='float32', overwrite=True)
	zarr_data.create_dataset('info_grasp', data=info_grasp_arrays, chunks=total_sub_chunk_size, dtype='float32', overwrite=True)
	zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('full_state', data=full_state_arrays, chunks=full_state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

	cprint(f'-'*50, 'cyan')
	# print shape
	cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
	cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
	cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
	cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
	cprint(f'full_state shape: {full_state_arrays.shape}, range: [{np.min(full_state_arrays)}, {np.max(full_state_arrays)}]', 'green')
	cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
	cprint(f'info_near shape: {len(info_near_arrays)}', 'green')
	cprint(f'info_grasp shape: {len(info_grasp_arrays)}', 'green')
	cprint(f'total_subs shape: {len(total_subs)}', 'green')
	cprint(f'Saved zarr file to {save_dir}', 'green')

	# clean up
	del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays, info_near_arrays,info_grasp_arrays,total_subs
	del zarr_root, zarr_data, zarr_meta
	del e

def qpos_plot(save_dir,n_groups,qpos_sub,apos_sub,apos_gripper_sub,qpos_gripper_sub,target_pos_sub,episode_idx):
	tstep = np.linspace(0, 1, len(qpos_sub)-1) 
	fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)
	save_path = os.path.join(save_dir,'plot',f'rollout{episode_idx}_qpos.png')
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	for n, ax in enumerate(axes):
		ax.plot(tstep, np.array(apos_sub)[1:, n], label=f'real qpos {n}')
		ax.plot(tstep, np.array(qpos_sub)[1:, n], label=f'sim target qpos {n}')
		ax.plot(tstep, np.array(target_pos_sub)[1:, n], label=f'target qpos {n}')
		ax.set_title(f'qpos {n}')
		ax.legend()

		plt.xlabel('timestep')
		plt.ylabel('qpos')
		plt.tight_layout()
		fig.savefig(os.path.join(save_dir, f"plot/rollout{episode_idx}_qpos.png"))
		plt.close(fig)
	tstep = np.linspace(0, 1, len(qpos_sub)-1) 
	apos_gripper_sub = np.expand_dims(apos_gripper_sub, axis=1)
	n_groups = 1
	fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)
	save_path = os.path.join(save_dir, 'plot', f'rollout{episode_idx}_qpos_gripper.png')
	os.makedirs(os.path.dirname(save_path), exist_ok=True)

	# 检查 n_groups，处理 axes 是单个对象的情况
	if n_groups == 1:
		# 如果只有一个 subplot，axes 是单个对象，不是列表
		axes.plot(tstep, np.array(apos_gripper_sub)[1:], label=f'real qpos')
		axes.plot(tstep, np.array(qpos_gripper_sub)[1:], label=f'sim target qpos')

		axes.set_title(f'qpos')
		axes.legend()

		plt.xlabel('timestep')
		plt.ylabel('qpos')
		plt.tight_layout()
		fig.savefig(save_path)
		plt.close(fig)
 
if __name__ == "__main__":
	# wandb.init(project="gen_demonstration_expert", name="anny-orange")
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='basketball')
	parser.add_argument('--num_episodes', type=int, default=10)
	parser.add_argument('--root_dir', type=str, default="../../3D-Diffusion-Policy/data/" )

	args = parser.parse_args()
	main(args)