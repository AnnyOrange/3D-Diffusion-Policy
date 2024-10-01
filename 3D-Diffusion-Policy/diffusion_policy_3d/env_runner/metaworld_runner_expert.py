import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
# from third_party/Metaworld/metaworld.policies import *
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(project_root, 'third_party/Metaworld'))
from metaworld.policies import *
class MetaworldRunner_expert(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=10,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        print(self.task_name)

        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, save_video=True):
        # device = policy.device
        device = "cuda:0"
        # dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env
        env_name = self.task_name
        mw_policy = self.load_mw_policy(env_name)
        print(mw_policy)
        
        for episode_idx in range(self.eval_episodes):
            
            # start rollout
            # import pdbpdb.set_trace()
            obs = env.reset()
            raw_state = obs['full_state']
            raw_state = raw_state.flatten()
            obs_dict = env.get_visual_obs()
            # obs_dict = env.get_visual_obs()
            # policy.reset()
            # raw_state = e.reset()['full_state']
            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # with torch.no_grad():
                obs_dict_input = {}
                obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                # action_dict = policy.predict_action(obs_dict_input)
                action = mw_policy.get_action(raw_state)
                    # print(len(action))

                # np_action_dict = dict_apply(action_dict,
                #                             lambda x: x.detach().to('cpu').numpy())
                # action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step([action])
                # print("info['success']",info['success'])

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        # _ = env.reset()
        videos = None

        return log_data
    def load_mw_policy(self,task_name):
        if task_name == 'peg-insert-side':
            agent = SawyerPegInsertionSideV2Policy()
        else:
            task_name = task_name.split('-')
            task_name = [s.capitalize() for s in task_name]
            task_name = "Sawyer" + "".join(task_name) + "V2Policy"
            agent = eval(task_name)()
        return agent