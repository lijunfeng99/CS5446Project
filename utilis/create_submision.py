import io
from utilis.Double_Qlearning_agent import Double_Qlearning_agent as SubmitAgent

import numpy as np
import random

def weights_to_string(model):
    weight_map = {
        "0.weight" : "fc1_weight",
        "0.bias" : "fc1_bias",
        "3.weight" : "fc2_weight",
        "3.bias" : "fc2_bias",
        "6.weight" : "fc3_weight",
        "6.bias" : "fc3_bias"
    }
    buffer = io.StringIO()
    for name, param in model.named_parameters():
        param_str = np.array2string(param.data.cpu().numpy(), threshold=np.inf, separator=', ')
        buffer.write(f"    weights[\"{weight_map[name]}\"]={param_str}\n\n")
    return buffer.getvalue()



def write_to_submission(model_name):
    agent = SubmitAgent(model_name)
    agent.load()
    code = f"""
def my_agent(observation, configuration):
    import numpy as np
    import random
    weights = {{}}
    def forward(x, weights):
        x = np.dot(x, weights['fc1_weight'].T) + weights['fc1_bias']
        x = np.maximum(0, x)  # ReLU激活
        x = np.dot(x, weights['fc2_weight'].T) + weights['fc2_bias']
        x = np.maximum(0, x)  # ReLU激活
        x = np.dot(x, weights['fc3_weight'].T) + weights['fc3_bias']
        return x
{weights_to_string(agent.model)}
    x = np.array(observation.board)
    for k, v in weights.items():
        weights[k] = np.array(v)
    q_values = forward(x, weights)
    action = np.argmax(q_values).item()
    if observation.board[action] > 0:
        return random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])
    return action
"""
    with open("submission.py", "w") as f:
        f.write(code)

if __name__ == "__main__":
    write_to_submission("DDQN_v1")
    pass