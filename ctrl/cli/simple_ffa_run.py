import warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from ctrl.agents import TensorForcePpoAgent

def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        # agents.SimpleAgent(),
        TensorForcePpoAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    for agent in agent_list:
        if type(agent) == TensorForcePpoAgent:
            agent.initialize(env)
            break

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} {} {} finished'.format(i_episode, reward, info))
    env.close()

if __name__ == '__main__':
    main()
