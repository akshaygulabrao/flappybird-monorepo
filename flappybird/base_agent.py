class BaseAgent:
    def decide(self, obs):
        raise NotImplementedError("Subclass must implement decide method")
