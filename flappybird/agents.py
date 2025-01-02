from pathlib import Path

import stable_baselines3


class BaseAgent:
    def decide(self, obs):
        raise NotImplementedError("Subclass must implement decide method")

class HandcraftedAgent(BaseAgent):
    def __init__(self, normalize:bool=True):
        self.normalize = normalize
        self.name = "handcrafted"

    def decide(self, obs):
        if not self.normalize:
            pipe = 0
            if obs[0] < 5:
                pipe = 1
            x = obs[pipe *3]
            bot = obs[pipe * 3 + 2]
            top = obs[pipe * 3 + 1]
            y_next = obs[-3] + obs[-2] + 24 + 1
            if 74 < x < 88 and obs[-3] - 45 >= top:
                return 1
            elif y_next >= bot:
                return 1
            return 0
        else:
            pipe = 0
            if obs[0] < 5/288:
                pipe = 1
            x = obs[pipe *3]
            bot = obs[pipe * 3 + 2]
            top = obs[pipe * 3 + 1]

            y_next = obs[-3] + (obs[-2]*10) / 512 # current y + current y_velocity
            y_next += 1/512 # 1 pixel acceleration per frame
            y_next += 24/512 # height of bird
            if 72/288 < x < 88/288 and obs[-3] - 45/512 >= top:
                    return 1
            elif y_next >= bot:
                return 1
            return 0


class DQNAgent(BaseAgent):
    def __init__(self, model_path:Path):
        self.model = stable_baselines3.DQN.load(model_path)
        self.name = model_path.stem

    def decide(self, obs):
        return int(self.model.predict(obs, deterministic=True)[0])
