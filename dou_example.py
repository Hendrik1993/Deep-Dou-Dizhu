''' An example of learning a Deep-Q Agent on Dou Dizhu
'''

import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
import timeit

from agents.ppo_agent import PPOAgent

# Make environment
env = rlcard.make('doudizhu', config={'seed': 0})
eval_env = rlcard.make('doudizhu', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 10
evaluate_num = 100
episode_num = 100
train_steps = 100

# The intial memory size
memory_init_size = 256 * 4

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/doudizhu_ppo_result/'

# Set a global seed
set_global_seed(0)

with tf.Session() as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent = PPOAgent(sess,
                     scope='ppo',
                     action_num=env.action_num,
                     # replay_memory_init_size=memory_init_size,
                     # train_every=train_every,
                     replay_memory_size=memory_init_size,
                     state_shape=env.state_shape,
                     batch_size=256,
                     mlp_layers=[512, 512])

    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents([agent, random_agent, random_agent])
    eval_env.set_agents([agent, random_agent, random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):

        print("EPISODE: ", episode)

        agent.memory.reset()

        while not agent.memory.is_full():

            # Generate data from the environment
            print("Collecting trajectories")
            #start_time = timeit.default_timer()
            trajectories, _ = env.run(is_training=False)
            #print("Time Taken: ", timeit.default_timer() - start_time)
            #print(f"Number of steps in game: Agent: {len(trajectories[0])}, Rnd1: {len(trajectories[1])}, Rnd2: {len(trajectories[2])}")
            #print("-----------------------")

            # Feed transitions into agent memory, and train the agent
            #print("Feeding trajectories")
            for ts in trajectories[0]:
                agent.feed(ts)

        last_value = agent.actor_critic.predict('values', sess, trajectories[0][-1][0])[0][0]
        agent.memory.calculate_advantage_gae(last_value)

        print("Beginning training...")
        #losses = []
        for i in range(train_steps):
            agent.train()


        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            print("Evaluating...")
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('PPO')

    # Save model
    save_dir = 'models/doudizhu_ppo'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
