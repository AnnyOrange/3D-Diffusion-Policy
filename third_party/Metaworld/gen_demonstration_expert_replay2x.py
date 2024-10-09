# bash scripts/metaworld/gen_demonstration_expert_replay.sh reach 5
import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from termcolor import cprint
import copy
import imageio
from metaworld.policies import *
import matplotlib.pyplot as plt
import imageio
# import faulthandler
# faulthandler.enable()

# seed = np.random.randint(0, 100)
# def seed_(save_dir):
# 	save_dir_2 = os.path.join(save_dir,'seed.txt')
# 	with open(save_dir_2, 'r') as f:
# 		seed = int(f.read())
# 	print("seed",seed)
# 	return seed
seed = 73
def avg_speed(total_count,total_sub):
	return np.sum(total_sub)/total_count

def load_mw_policy(task_name):
	if task_name == 'peg-insert-side':
		agent = SawyerPegInsertionSideV2Policy()
	else:
		task_name = task_name.split('-')
		task_name = [s.capitalize() for s in task_name]
		task_name = "Sawyer" + "".join(task_name) + "V2Policy"
		agent = eval(task_name)()
	return agent

def load_mw_data(save_dir):
	zarr_root_dir = save_dir
	zarr_data_dir = zarr_root_dir + '/data'
	zarr_meta_dir = zarr_root_dir + '/meta'
	zarr_data = zarr.open(zarr_data_dir, mode='r')
	action = zarr_data['action'][:]
	# print(action.shape)
	info_near = zarr_data['info_near'][:]
	# print(info_near)
	info_grasp = zarr_data['info_grasp'][:]
	total_sub = zarr_data['total_sub'][:]
	target_pos_path = os.path.join(save_dir, "target_pos.npy")
	target_pos = np.load(target_pos_path)
	sim_pos_path = os.path.join(save_dir,"sim_pos.npy")
	sim_pos = np.load(sim_pos_path)
	return action,info_near,info_grasp,total_sub,target_pos,sim_pos

# 
def quit(idx,total_sub_):
	if idx<=total_sub_:
		return True
	return False

def save_video_to_file(videos, save_path, fps=10):
    if len(videos.shape) == 5:
        videos = videos[:, 0]  # 如果 shape 是 5 维，去掉 batch 维度
    # print(videos.shape)
    videos = np.transpose(videos, (0, 2, 3, 1)) 
    with imageio.get_writer(save_path, fps=fps) as video_writer:
        for frame in videos:
            video_writer.append_data(frame)
    # print(f"Video saved at {save_path}")
def target_to_action(target_pos,current_obs,scale):
	delta_action = (target_pos-current_obs)/scale
	return delta_action
def sim_to_action(sim_pos,current_obs,scale):
	delta_action = (sim_pos-current_obs)/scale
	return delta_action

def main(args):
	save_video = True
	env_name = args.env_name
	# method = args.method
	# speed = args.speed
	target_action = np.zeros(4)
	sim_action = np.zeros(4)
	save_dir_old = os.path.join(args.root_dir, 'metaworld_'+args.env_name+'_expert.zarr')
	save_dir = os.path.join(args.root_dir, 'metaworld_'+args.env_name+f'_expert_2xspeeddemo.zarr')
	seed = 73
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
	action_data,info_near_data,info_grasp_data,total_sub,target_pos_arrays,sim_pos_arrays = load_mw_data(save_dir_old)
	# print(len(target_pos_arrays))
	# import pdb;pdb.set_trace()

	e = SimpleVideoRecordingWrapper(MetaWorldEnv(env_name, device="cuda:0", use_point_crop=True, seed = seed))
	num_episodes = args.num_episodes
	cprint(f"Number of episodes : {num_episodes}", "yellow")
	

	total_count = 0
	img_arrays = []
	point_cloud_arrays = []
	depth_arrays = []
	state_arrays = []
	full_state_arrays = []
	action_arrays = []
	target_arrays = []
	episode_ends_arrays = []
	all_success_rates = []
	all_traj_rewards = []
	
	episode_idx = 0
	qpos_arrays = []
	apos_arrays = []
	apos_gripper_arrays = []
	qpos_gripper_arrays = []
	sim_qpos_arrays = []
	new_count = []	

	# mw_policy = load_mw_policy(env_name)
	# print(mw_policy)

	# loop over episodes
	raw_state_file = os.path.join(save_dir, 'raw_state.txt')
	with open(raw_state_file, 'a') as raw_state_log:
		action_idx = 0
		while episode_idx < num_episodes:
			raw_state = e.reset()['full_state']
			raw_state_log.write(f'{episode_idx}: raw_state = {raw_state.tolist()}\n')
			# TODO将raw_state 存成raw_state.txt 每行是episode_idx: raw_state = []

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
			total_count_sub = 0
			apos_sub = []
			qpos_sub = []
			apos_gripper_sub = []
			qpos_gripper_sub = []
			action_idx_episode_idx = 0
			
			while not done:
				if total_sub[episode_idx]<=action_idx_episode_idx:
					print("total_sub[episode_idx]",total_sub[episode_idx])
					print("total_count_sub",total_count_sub)
					print("action_idx",action_idx)
					break
				obs_img = obs_dict['image']
				obs_robot_state = obs_dict['agent_pos']
				obs_point_cloud = obs_dict['point_cloud']
				obs_depth = obs_dict['depth']
	

				img_arrays_sub.append(obs_img)
				point_cloud_arrays_sub.append(obs_point_cloud)
				depth_arrays_sub.append(obs_depth)
				state_arrays_sub.append(obs_robot_state)
				full_state_arrays_sub.append(raw_state)
				
				# action = mw_policy.get_action(raw_state)
				# qpos = sim_action[:3]/100+raw_state[:3]
				# qpos = action[:3]+raw_state[:3]
				# print("raw_state",raw_state[:3])
				apos_sub.append(raw_state[:3])
				# qpos_sub.append(qpos)
				apos_gripper_sub.append(raw_state[3])
				# qpos_gripper_sub.append(qpos_gripper)
				
				# import pdb;pdb.set_trace()
				# print(green_curve)
				
				
				action1 = action_data[action_idx]
				action_idx+=1
				action1 = np.clip(action1, -1, 1)
				action_idx_episode_idx+=1
				
				
				if total_sub[episode_idx]<=action_idx_episode_idx:
					print("total_sub[episode_idx]",total_sub[episode_idx])
					print("total_count_sub",total_count_sub)
					print("action_idx",action_idx)
					break
				action2 = action_data[action_idx]
				action_idx+=1
				action2 = np.clip(action2, -1, 1)
				action_idx_episode_idx+=1
				action_2x = action1+action2
				action_2x[-1] = action_data[action_idx-2][-1]
				# print(action_2x)
				action_arrays_sub.append(action_2x)
				green_curve=None
				obs_dict, reward, done, info = e.step(action_2x,green_curve)
				total_count_sub += 1
				# import pdb;pdb.set_trace()
				
				
				# print(info)
				# print("info",info)
				
				raw_state = obs_dict['full_state']
				ep_reward += reward
	
				ep_success = ep_success or info['success']
				ep_success_times += info['success']

				
				if done:
					print("done")
					print("total_sub",total_count_sub)
					# total_count+=total_count_sub
					break
			
			new_count.append(total_count_sub)
			all_success_rates.append(ep_success)
			all_traj_rewards.append(ep_reward)
			# qpos_arrays.append(qpos_sub)
			apos_arrays.append(apos_sub)
			apos_gripper_arrays.append(apos_gripper_sub)
			# qpos_gripper_arrays.append(qpos_gripper_sub)
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
				# episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
				# img_arrays.extend(copy.deepcopy(img_arrays_sub))
				# point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
				# depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
				# state_arrays.extend(copy.deepcopy(state_arrays_sub))
				# action_arrays.extend(copy.deepcopy(action_arrays_sub))
				# full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))
				episode_idx += 1
			else:
				total_count += total_count_sub
				print("total_count",total_count)
				print("len(img_arrays_sub)",len(img_arrays_sub))
				print("action_arrays_sub",len(action_arrays_sub))
				episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
				img_arrays.extend(copy.deepcopy(img_arrays_sub))
				point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
				depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
				state_arrays.extend(copy.deepcopy(state_arrays_sub))
				action_arrays.extend(copy.deepcopy(action_arrays_sub))
				full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))
				cprint('Episode: {}, Reward: {}, Success Times: {}'.format(episode_idx, ep_reward, ep_success_times), 'green')
				save_dir_3 = os.path.join(save_dir,'succ_reward_replay.txt')
				with open(save_dir_3, 'a') as f:
					f.write('Episode: {}, Reward: {}, Success Times: {}\n'.format(episode_idx, ep_reward, ep_success_times))
				episode_idx += 1
	
	log_data = dict()	
	log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
	log_data['mean_success_rates'] = np.mean(all_success_rates)	
	log_data['avg_speed'] = avg_speed(total_count,total_sub)
	# log_data['method'] = method
	log_file_path = os.path.join(save_dir, 'logdata.txt')
	with open(log_file_path, 'w') as f:
		for key, value in log_data.items():
			f.write(f"{key}: {value}\n")
	n_groups = 3
	# print(len(qpos_arrays))
	start = 0
	print(len(info['target_pos']))
	save_path = os.path.join(save_dir, 'target_pos.npy')
	np.save(save_path, info['target_pos'])
	for idx in range(num_episodes):
	# 	target_pos_sub = info['target_pos'][start:start+new_count[idx]]
	# 	sim_pos_sub = sim_qpos_arrays[start:start+new_count[idx]]
		qpos_plot(save_dir,n_groups,apos_arrays[idx],apos_gripper_arrays[idx],idx)
		start = start+new_count[idx]
	# videos = e.get_video()
	# if len(videos.shape) == 5:
	# 	videos = videos[:, 0] 
	
	# if save_video:
	# 	videos_wandb = wandb.Video(videos, fps=10, format="mp4")
	# 	log_data[f'sim_video_eval'] = videos_wandb
	# videos = None
	# log_file_path = os.path.join(save_dir, 'logdata.txt')
	# with open(log_file_path, 'w') as f:
	# 	for key, value in log_data.items():
	# 		f.write(f"{key}: {value}\n")
 
	# save data
 	###############################
    # save data
    ###############################
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
	# zarr_data.create_dataset('total_sub', data=total_subs, chunks=total_sub_chunk_size, dtype='int64', overwrite=True)
	# zarr_data.create_dataset('info_near', data=info_near_arrays, chunks=total_sub_chunk_size, dtype='float32', overwrite=True)
	# zarr_data.create_dataset('info_grasp', data=info_grasp_arrays, chunks=total_sub_chunk_size, dtype='float32', overwrite=True)
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
	# cprint(f'info_near shape: {len(info_near_arrays)}', 'green')
	# cprint(f'info_grasp shape: {len(info_grasp_arrays)}', 'green')
	# cprint(f'total_subs shape: {len(total_subs)}', 'green')
	cprint(f'Saved zarr file to {save_dir}', 'green')

	# clean up
	del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
	del zarr_root, zarr_data, zarr_meta
	del e

def qpos_plot(save_dir,n_groups,apos_sub,apos_gripper_sub,episode_idx):
	tstep = np.linspace(0, 1, len(apos_sub)-1) 
	fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)
	save_path = os.path.join(save_dir,'plot',f'rollout{episode_idx}_qpos.png')
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	for n, ax in enumerate(axes):
		ax.plot(tstep, np.array(apos_sub)[1:, n], label=f'real qpos {n}')
		# ax.plot(tstep, np.array(qpos_sub)[1:, n], label=f'sim target qpos {n}')
		# ax.plot(tstep, np.array(target_pos_sub)[1:, n], label=f'target qpos {n}')
		# ax.plot(tstep, np.array(sim_pos_sub)[1:, n], label=f'save sim target qpos {n}')
		ax.set_title(f'qpos {n}')
		ax.legend()

		plt.xlabel('timestep')
		plt.ylabel('qpos')
		plt.tight_layout()
		fig.savefig(os.path.join(save_dir, f"plot/rollout{episode_idx}_qpos.png"))
		plt.close(fig)
	tstep = np.linspace(0, 1, len(apos_sub)-1) 
	apos_gripper_sub = np.expand_dims(apos_gripper_sub, axis=1)
	n_groups = 1
	fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)
	save_path = os.path.join(save_dir, 'plot', f'rollout{episode_idx}_qpos_gripper.png')
	os.makedirs(os.path.dirname(save_path), exist_ok=True)

	# 检查 n_groups，处理 axes 是单个对象的情况
	if n_groups == 1:
		# 如果只有一个 subplot，axes 是单个对象，不是列表
		axes.plot(tstep, np.array(apos_gripper_sub)[1:], label=f'real qpos')
		# axes.plot(tstep, np.array(qpos_gripper_sub)[1:], label=f'sim target qpos')

		axes.set_title(f'qpos')
		axes.legend()

		plt.xlabel('timestep')
		plt.ylabel('qpos')
		plt.tight_layout()
		fig.savefig(save_path)
		plt.close(fig)
 
if __name__ == "__main__":
	# wandb.init(project="gen_demonstration_expert_replay", name="anny-orange")
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='basketball')
	parser.add_argument('--num_episodes', type=int, default=10)
	parser.add_argument('--root_dir', type=str, default="../../3D-Diffusion-Policy/data/" )
	parser.add_argument('--method', type=int,default=0)
	parser.add_argument('--speed', type=int, default = 2)

	args = parser.parse_args()

	main(args)
