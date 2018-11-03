import warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py \
 --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
 --config=PommeFFACompetition-v0
"""

import atexit
import functools
import os, sys

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym

sys.path.append('.')

from pommerman import helpers, make
from pommerman.agents import TensorForceAgent
from ctrl.agents import TensorForcePpoAgent

CLIENT = docker.from_env()

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]

class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize
        self.prev_action = None
        self.curr_action = None
        self.old_position = None
        self.prev_position = None
        self.curr_position = None
        self.timestep = 0
        self.episode = 0
        self.has_blast_strength = False
        self.has_can_kick = False
        self.has_ammo = False
        print(f'Episode[{self.episode:03}], Timestep[{self.timestep:03}] initialized.')
    
    rule_of_thumb = """
        1. 승리하면 +300
        2. 패배하면 -150
        3. 아무 행동도 하지 않으면 타임 스텝마다 -0.2
        4. 어떤 행동이라도 하면 타임스텝마다 +0.1
         (구석에서 짱박혀서 움직이는 척 하기 있기 없기)
        5. 어떤 행동 중에서 폭탄은 +10
        6. 어떤 아이템이라도 먹으면 +20
    """
    def shaping_reward(self, agent_id, agent_obs, agent_reward, agent_action):
        import numpy as np
        self.timestep += 1
        self.agent_board = agent_obs['board']
        self.curr_position = np.where(self.agent_board == agent_id)

        modified_reward = agent_reward
        if agent_reward == 1:
            modified_reward += 300
        if agent_reward == -1:
            modified_reward -= 150
        if self.prev_position != None and self.prev_position == self.curr_position and self.old_position == self.prev_position:
            modified_reward -= 0.2
        if self.prev_position != None and self.prev_position != self.curr_position and self.old_position == self.prev_position:
            modified_reward += 0.1
        if self.curr_action == 5:
            modified_reward += 10
        if not self.has_blast_strength and int(agent_obs['blast_strength']) > 2:
            modified_reward += 20
            self.has_blast_strength = True
        if not self.has_can_kick and agent_obs['can_kick'] == True:
            modified_reward += 20
            self.has_can_kick = True
        if not self.has_ammo and int(agent_obs['ammo']) > 1:
            modified_reward += 20
            self.has_ammo = True

        print(f'Episode[{self.episode:03}], Timestep[{self.timestep:03}] got reward {modified_reward}')
        self.old_position = self.prev_position
        self.prev_position = self.curr_position
        return modified_reward

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])

        agent_id = self.gym.training_agent + 10
        agent_reward = reward[self.gym.training_agent]
        agent_action = all_actions[self.gym.training_agent]
        agent_obs = obs[self.gym.training_agent]
        modified_reward = self.shaping_reward(agent_id, agent_obs, agent_reward, agent_action)

        return agent_state, terminal, modified_reward

    '''Reset method is called when every episode starts'''
    def reset(self):
        print(f'Episode[{self.episode:03}], Timestep[{self.timestep:03}] ended.\n')
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        self.timestep = 0
        self.episode += 1
        self.has_blast_strength = False
        self.has_can_kick = False
        self.has_ammo = False
        return agent_obs

def create_ppo_agent(agent):
    if type(agent) == TensorForceAgent:
        print("create_ppo_agent({})".format(agent))
        return TensorForcePpoAgent()
    return agent

def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo,test::agents.SimpleAgent,"
        "test::agents.SimpleAgent,test::agents.SimpleAgent",
        help="Comma delineated list of agent types and docker "
        "locations to run the agents.")
    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
        "to pass to Docker. This is only for the Docker Agent."
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        "would send two arguments to Docker Agent 0 and one to"
        " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
        "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
        "None.")
    parser.add_argument(
        "--checkpoint",
        default="models/ppo",
        help="Directory where checkpoint file stored to."
    )
    parser.add_argument(
        "--num_of_episodes",
        default="10",
        help="Number of episodes"
    )
    parser.add_argument(
        "--max_timesteps",
        default="2000",
        help="Number of steps"
    )
    args = parser.parse_args()

    config = args.config
    # record_pngs_dir = args.record_pngs_dir
    # record_json_dir = args.record_json_dir
    # agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file
    checkpoint = args.checkpoint
    num_of_episodes = int(args.num_of_episodes)
    max_timesteps = int(args.max_timesteps)

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        create_ppo_agent(helpers.make_agent_from_string(agent_string, agent_id + 1000))
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    env = make(config, agents, game_state_file)
    training_agent = None
    training_agent_id = None

    for agent in agents:
        if type(agent) == TensorForcePpoAgent:
            print("Ppo agent initiazlied : {}, {}".format(agent, type(agent)))
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            training_agent_id = agent.agent_id
            break
        print("[{}] : id[{}]".format(agent, agent.agent_id))

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env)

    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=args.render)
    runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=num_of_episodes, max_episode_timesteps=max_timesteps)
    print("Stats: ",
        runner.episode_rewards[-30:],
        runner.episode_timesteps,
        runner.episode_times)

    agent.save_model(checkpoint)

    rewards = runner.episode_rewards
    import numpy as np
    mean = np.mean(rewards)
    print('last 30 rewards {}'.format(rewards[-30:]))
    print('mean of rewards {}'.format(mean))

    try:
        runner.close()
    except AttributeError as e:
        print(e)
        pass

if __name__ == "__main__":
    main()