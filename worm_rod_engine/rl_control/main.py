import os
import gymnasium as gym
import torch
import sys
import time
import datetime
import pickle

from stable_baselines3 import PPO # [PPO, SAC] Import RL agent.
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

print('Finished importing python libraries.')

from parameters import parameters
from save_results import save_results

print('Finished importing files.')

main_dir = os.path.dirname(__file__)
DEBUG = True


# Set animal model
if(len(sys.argv) >= 2): # If an animal name is provided.
    animal = sys.argv[1]
else:
    animal = None

# Set directories and parameters
if(len(sys.argv) >= 3): # If an existing model is provided.
    path_split = os.path.split(sys.argv[2])
    model_path = path_split[0]
    model_file = path_split[1]
    
    # Set parameters
    if(os.path.exists(f'{model_path}/parameters.pkl')): # If an existing parameters file is provided.
        with open(f'{model_path}/parameters.pkl', 'rb') as f:
            P = pickle.load(f)
        
        from training_params import training_params
        P["training"] = training_params(use_gpu=False, T=P["T"], dt=P["dt_opt"], animal=P['animal'])
    else:
        P = parameters()
else: # Create a new model.
    P = parameters()
    
    direction = "forward" if P["direction"] == 1 else "backward"
    name = f"{P['animal']}_{P['gait']}_{direction}_T={str(P['T'])}_dt={str(P['dt'])}_dt-opt={str(P['dt_opt'])}"
    
    current_datetime = datetime.datetime.now()
    model_file = f"model_{name}_{current_datetime.strftime('%Y%m%d%H%M%S')}"
    model_path = f"resources/results/{P['animal']}/{model_file}/"

def make_env():
    model_path = os.path.join(main_dir, 'envs', 'mujoco', 'assets', P["physical_model"]["file_name"] + ".xml")
    return gym.make(P["env"], P=P, max_episode_steps=P["training"]["steps_per_episode"])
    
def main():
    env, model = get_env_and_model()

    # register(
    #     id='my-env-v0',  # Unique identifier
    #     entry_point='my_module:MyEnv',  # Where to find the Env class
    #     max_episode_steps=1000,  # Default episode length
    #     kwargs={'param': value}  # Default parameters
    # )

    for i in range(P["training"]["batch_num"]):
        if(P["training"]["device"] == "cuda"):
            model.learn(total_timesteps=P["training"]["steps_per_batch"], tb_log_name=name, reset_num_timesteps=(i==0))
            model.save(model_path + os.sep + model_file + '.zip') # Save the model.
            print("Done training. Model Saved.")


    with open(f'{model_path}/parameters.pkl', 'wb') as f:
        pickle.dump(P, f)
    
    env.close()
    save_results(model=model, P=P, model_path=model_path, model_file=model_file, ep=i)
    
def get_env_and_model():
    
    # Set RL environment(s)

    if DEBUG:
        env = DummyVecEnv([make_env for _ in range(P["training"]["n_envs"])])
    else:
        env = SubprocVecEnv([make_env for _ in range(P["training"]["n_envs"])])
    
    env = VecMonitor(env, P["training"]["tensorboard_log_path"])
    print("Action space:", env.action_space)
    print("Obs space:", env.observation_space)
    
    # Set model
    if(len(sys.argv) >= 3): # If an existing model is provided.
        model = PPO.load(model_path + os.sep + model_file, env=env) # Load the saved model.
        print("Loading an existing PPO model.")
    else: # Create new model.
        policy_kwargs = dict(activation_fn=P["training"]["activation_function"], net_arch=dict(pi=P["training"]["policy_network"], vf=P["training"]["value_network"]))
        
        model = PPO("MlpPolicy", env, gamma=P["training"]["gamma"], ent_coef=P["training"]["ent_coef"], device=P["training"]["device"], policy_kwargs=policy_kwargs, verbose=P["training"]["verbose"], tensorboard_log=P["training"]["tensorboard_log_path"])
        
        print("Creating a new model and training a PPO agent.")
    # print("Policy:", model.policy)
    
    return env, model

if __name__ == '__main__':
	main()