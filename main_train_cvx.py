import datetime
from train_cvx import train

SACConfig = {
    "env_name": "MEC_OFFLOAD_SERVICE_BANDWITH_RESOURCE_ALLOCATE",
    "hidden_units": [512, 512],  
    "seed": 123,  
    "directory": "models/",
    "sub_directory": None,
    "batch_size": 256,  
    "memory_size": 1000000,
    "learning_rate": 3e-5,  
    "alpha_learning_rate": 3e-5,
    "gamma": 0.993,  
    "tau": 0.005,
    "num_steps": 1000001,  
    "alpha": 0.05,
    "num_task": 30,  
    "ap_storage_capacity": 300,  
    "bandwith": 20,  
    "ap_computing_capacity": 20, 
    "cloud_computing_rate": 5,  
    "ap2c_rate": 8,
}


def main(mode, **kwargs):
    if mode == "train":
        print("Training SAC with the following arguments:")
        print(kwargs)
        train(**kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    main("train", **SACConfig)
