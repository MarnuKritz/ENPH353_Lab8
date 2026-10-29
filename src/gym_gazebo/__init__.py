from .core.gazebo_env import GazeboEnv
from .envs.linefollower_env import LineFollowerEnv
from gymnasium.envs.registration import register
from .envs.cartpole_env import CartpoleEnv

register(
    id='LineFollowerEnv-v0',                                      
    entry_point='gym_gazebo.envs.linefollower_env:LineFollowerEnv',
    max_episode_steps=3000,                                   
)

register(
    id='CartpoleEnv-v0',                                      
    entry_point='gym_gazebo.envs.cartpole_env:CartpoleEnv',
    max_episode_steps=3000,                                   
)