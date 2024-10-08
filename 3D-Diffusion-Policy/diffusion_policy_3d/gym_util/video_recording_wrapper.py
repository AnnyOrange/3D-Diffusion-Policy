import gym
import numpy as np
from termcolor import cprint


class SimpleVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            steps_per_render=1,
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        print("SimpleVideoRecordingWrapper",env)
        self.mode = mode
        self.steps_per_render = steps_per_render

        self.step_count = 0

    def reset(self, **kwargs):
        # import traceback
        # print("Reset called")
        # traceback.print_stack()  
        obs = super().reset(**kwargs)
        self.frames = list()

        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        self.step_count = 1
        return obs
    
    def step(self, action,green_curve=None):
        if green_curve is None:
            result = super().step(action)
        else:
            result = super().step(action,green_curve)
        # result = super().step(action,green_curve)
        # result = super().step(action)
        self.step_count += 1
        
        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result
    
    def get_video(self):
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        return video

