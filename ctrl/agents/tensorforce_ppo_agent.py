"""
A Work-In-Progress agent using Tensorforce
"""
import os
import numpy as np

# from pommerman.agents import BaseAgent
from pommerman import characters
from ctrl.agents import BaselineAgent

class TensorForcePpoAgent(BaselineAgent):
# class TensorForcePpoAgent(BaseAgent):
    """The TensorForcePpoAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber, algorithm='ppo', checkpoint='models/checkpoint'):
        super(TensorForcePpoAgent, self).__init__(character)
        self.algorithm = algorithm
        self.checkpoint = checkpoint
        self.agent = None
        self.state = {}
        self.env = None
        self.version = self.reload_version()
        print("TensorForcePpoAgent {} iniitialized.".format(self.version))

    def reload_version(self, filename='VERSION'):
        version = None
        for line in open(filename, 'r'):
            version = line.strip().split('=')[1]
            break
        return version
        
    def episode_end(self, reward):
        # print("i've got rewards {}".format(reward))
        pass

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions. See train_with_tensorforce."""
        print("obs '{}'".format(obs))
        agent_state = self.env.featurize(obs)
        print("featureize '{}'".format(agent_state))
        action = self.agent.act(agent_state)
        return action

    def initialize(self, env):
        from gym import spaces
        from tensorforce.agents import PPOAgent
        self.env = env

        # activation function 이 없으므로 depth 가 깊어지면 decay 문제.
        network_spec = [
            dict(type='dense', size=64),
            dict(type='dense', size=64)
        ]

        summarizer = dict(
            directory="board",
            steps=50,
            labels=[
                "graph", 
                "losses",
                "total-loss",
                "variables",
                "inputs",
                "states",
                "actions",
                "rewards",
                "gradients",
                "gradients_histogram",
                "gradients_scalar",
                "regularization"
                # "configuration"
            ]
        )

        if self.algorithm == "ppo":
            if type(env.action_space) == spaces.Tuple:
                actions = {
                    str(num): {
                        'type': int,
                        'num_actions': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
            else:
                actions = dict(type='int', num_actions=env.action_space.n)

            self.agent = PPOAgent(
                states=dict(type='float', shape=env.observation_space.shape),
                actions=actions,
                network=network_spec,
                summarizer=summarizer,
                # Agent
                states_preprocessing=None,
                actions_exploration=None,
                reward_preprocessing=None,
                # MemoryModel
                update_mode=dict(
                    unit='episodes',
                    # 100 episodes per update
                    batch_size=100,
                    # Every 10 episodes
                    frequency=10
                ),
                memory=dict(
                    type='latest',
                    include_next_states=False,
                    capacity=5000
                ),
                # DistributionModel
                distributions=None,
                entropy_regularization=0.01,
                # PGModel
                baseline_mode='states',
                baseline=dict(
                    type='mlp',
                    sizes=[64, 64]
                ),
                baseline_optimizer=dict(
                    type='multi_step',
                    optimizer=dict(
                        type='adam',
                        learning_rate=1e-3
                    ),
                    num_steps=5
                ),
                gae_lambda=0.97,
                # PGLRModel
                likelihood_ratio_clipping=0.2,
                # PPOAgent
                step_optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                subsampling_fraction=0.2,
                optimization_steps=25,
                execution=dict(
                    type='single',
                    session_config=None,
                    distributed_spec=None
                )
            )
                # batching_capacity=1000,
                # step_optimizer=dict(type='adam', learning_rate=1e-4))

            self.restore_model_if_exists(self.checkpoint)

        return self.agent

    def restore_model_if_exists(self, checkpoint):
        if os.path.isfile(checkpoint):
            pardir = os.path.abspath(os.path.join(checkpoint, os.pardir))
            self.agent.restore_model(pardir)
            print("tensorforce model '{}' restored.".format(pardir))

    def save_model(self, checkpoint):
        pardir = os.path.abspath(os.path.join(checkpoint, os.pardir))
        if not os.path.exists(pardir):
            os.mkdir(pardir)
            print("checkpoint dir '{}' created.".format(pardir))
        checkpoint_path = self.agent.save_model(pardir, False)
        print("checkpoint model '{}' saved.".format(checkpoint_path))

