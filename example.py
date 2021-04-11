"""
Toy example taken from RLcard's github
"""

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, print_card

from utils.utils import get_action_names, get_card_names

# Make environment
env = rlcard.make('uno', config={'seed': 0})
episode_num = 1

# Set a global seed
set_global_seed(2)

# Set up agents
"""
This actually does not make sense to me, since apparently ín this implementation only two players can play (Tom)
"""
agent_0 = RandomAgent(action_num=env.action_num)
agent_1 = RandomAgent(action_num=env.action_num)
agent_2 = RandomAgent(action_num=env.action_num)
agent_3 = RandomAgent(action_num=env.action_num)
env.set_agents([agent_0, agent_1, agent_2, agent_3])

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=False)
    print(env.player_num)
    # Print out the trajectories
    print('\nEpisode {}'.format(episode))

    for ts in trajectories[0]:
        #print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.format(ts[0], get_action_names(ts[1]), ts[2], ts[3], ts[4]))
        # print("STATE:")
        # print(f"Hand Player 1: {get_card_names(ts[0]['obs'][1])}")
        # print(f"Action Player 1: {get_action_names(ts[1])}")
        # #print(f"Action: {get_action_names(ts[1])}")
        # print(f"Hand Player 2: {get_card_names(ts[0]['obs'][5])}")
        # print(f"Target: {get_card_names(ts[0]['obs'][3])}")
        #
        # print("NEW STATE")
        # print(f"Hand Player 1: {get_card_names(ts[3]['obs'][1])}")
        #
        # # print(f"Action: {get_action_names(ts[1])}")
        # print(f"Hand Player 2: {get_card_names(ts[3]['obs'][5])}")
        # print(f"Target: {get_card_names(ts[3]['obs'][3])}")
        # # print(f"Target: {get_card_names(ts[0]['obs'][3])}")
        # # print(f"Action: {get_action_names(ts[1])}")

        print(env.get_perfect_information())
