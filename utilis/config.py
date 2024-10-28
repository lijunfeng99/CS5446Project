import os
import json
CONFIG_PATH = "./config/"


class Config(object):
    def __init__(self, MODEL_NAME= "", auto_load=False):
        # The environment config
        self.input_shape = 42
        self.num_actions = 7
        self.env_name = "connectx"

        # The train config
        self.MODEL_NAME = MODEL_NAME
        self.INVALID_MOVE_PENALTY = -2
        self.NUM_EPISODES = 100

        self.EPSILON_START = 0.9
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 0.99
        self.TARGET_UPDATE = 1000

        self.GAMMA = 0.99

        self.DEVICE = "mps" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if auto_load:
            self.load()
    
    def save(self):
        with open(os.path.join(CONFIG_PATH, self.MODEL_NAME + ".json"), 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        print("Config saved to", os.path.join(CONFIG_PATH, self.MODEL_NAME + ".json"))

    def load(self):
        with open(os.path.join(CONFIG_PATH, self.MODEL_NAME + ".json"), 'r') as f:
            config_dict = eval(f.read())
            self.__dict__.update(config_dict)
        print("Config loaded from", os.path.join(CONFIG_PATH, self.MODEL_NAME + ".json"))
    
    def rename(self, new_name):
        self.MODEL_NAME = new_name
        self.save()