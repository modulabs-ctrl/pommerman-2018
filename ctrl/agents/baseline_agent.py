'''This is the base abstraction for agents in pommerman.
All agents should inherent from this class'''
from pommerman import characters
from pommerman.agents import BaseAgent

class BaselineAgent(BaseAgent):
    prev_action = None
    curr_action = None

    def __init__(self, character=characters.Bomber):
	    super(BaselineAgent, self).__init__(character)

    def restore_model_if_exists(self, checkpoint):
        pass

    def save_model(self, checkpoint):
        pass

    def get_prev_action(self):
        return None

    def get_curr_action(self):
        return None

