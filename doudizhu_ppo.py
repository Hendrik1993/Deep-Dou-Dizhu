"""
Training and Evaluating a PPO Agent on Dou Dizhu
"""

import tensorflow as tf
import os
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from agents.ppo_agent import PPOAgent


# Make environment
env = rlcard.make('doudizhu', config={'seed': 0})
eval_env = rlcard.make('doudizhu', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 1
evaluate_num = 100
episode_num = 100
train_steps = 10

# The intial memory size
memory_init_size = 8192

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
                     replay_memory_size=memory_init_size,
                     state_shape=env.state_shape,
                     batch_size=64,
                     mlp_layers=[512, 512])

    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents([agent, random_agent, random_agent])
    eval_env.set_agents([agent, random_agent, random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)
    game_count = 0

    for episode in range(episode_num):

        print("EPISODE: ", episode)

        agent.memory.reset()

        while not agent.memory.is_full():

            # Generate data from the environment
            trajectories, _ = env.run(is_training=False)
            game_count += 1

            # Feed transitions into agent memory
            for ts in trajectories[0]:
                agent.feed(ts)

        # When memory full, calculate advantages. Therefore, add last value estimate
        last_value = agent.actor_critic.predict('values', sess, trajectories[0][-1][-2])[0][0]
        agent.memory.calculate_advantage_gae(last_value, trajectories[0][-1][-1])

        print("Beginning training...")
        for i in range(train_steps):
            loss, critic_loss, actor_loss = agent.train()



        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            print("Evaluating...")
            eval_env.set_agents([agent, random_agent, random_agent])
            logger.log_performance(game_count, tournament(eval_env, evaluate_num)[0])


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
