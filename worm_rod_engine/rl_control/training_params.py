import torch

def training_params(use_gpu=False, T=0.5, dt=0.01, animal="animal"):
    
    # dt is the optimisation time step (i.e., the physics time step x frame_skip).
    
    P = {
        "policy_network": [128, 128], # [[64,64], [128,128]] default in PPO. [256, 256].
        "value_network": [128, 128], # [[64,64], [128,128]] default in PPO. [256, 256].
        "activation_function": torch.nn.Tanh, # `torch.nn.ReLU` (default in SAC). `torch.nn.Tanh` (default in PPO).
        "algorithm": "PPO", # PPO, SAC, TD3, DDPG.
        "gamma": 0.99, # Default=*0.99*. [0.1, 0.8].
        "ent_coef": 0, # Default=0.0. [*0.001*, Infinity=*0.0005*].
        
        "verbose": 1,
        "tensorboard_log_path": f"resources/results/{animal}/tensorboard_log/",
    }
    
    # use_gpu = False
    if(use_gpu and torch.cuda.is_available()):
        P["device"] = "cuda"
        P["n_envs"] = 1
        
        P["total_timesteps"] = 6*(10**6) # / (dt / 0.01)
        P["steps_per_batch"] = P["total_timesteps"]
        P["batch_num"] = int(P["total_timesteps"] / P["steps_per_batch"])
        
        print("Using GPU")
    else:
        P["device"] = "cpu"
        P["n_envs"] = 1
        
        P["total_timesteps"] = 10*(10**6)
        P["steps_per_batch"] = 1*(10**6)
        P["batch_num"] = 1
        
        print("Using CPU")
        
    P["steps_per_period"] = round(T / dt)
    P["steps_per_episode"] = 1 * P["steps_per_period"] + 1
    
    return P