"""
Implementation of our PPO Agent
"""
import tensorflow as tf
from rlcard.utils import remove_illegal
import random
from dataclasses import dataclass, astuple
from scipy.stats import entropy
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool
    advantage: float
    discounted_reward: float

    def __iter__(self):
        return iter(astuple(self))


class PPOAgent(object):

    def __init__(self,
                 sess,
                 scope,
                 discount_factor=0.99,
                 epsilon_decay_steps=20000,
                 replay_memory_size=20000,
                 batch_size=32,
                 action_num=2,
                 state_shape=None,
                 learning_rate=0.00005,
                 mlp_layers=None,
                 training_steps=20):
        self.state_shape = state_shape
        self.action_num = action_num
        self.use_raw = False
        self.sess = sess
        self.scope = scope
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.clip_param = 0.2
        self.training_steps = training_steps

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        self.actor_critic = Estimator(scope="ActorCriticNet", action_num=self.action_num, value_num=1,
                                      learning_rate=learning_rate,
                                      state_shape=self.state_shape, mlp_layers=mlp_layers, output_func=tf.nn.softmax,
                                      clip_param=self.clip_param)
        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

    def feed(self, ts):
        """
        Decode tuple returned from the environments and feed it to the memory
        :param ts: Tuple of transitions
        """
        (state, action_info, reward, next_state, done) = tuple(ts)

        action, log_prob, value, entropy = action_info

        self.feed_memory(state, action, reward, log_prob, value, done)

    def feed_memory(self, state, action, reward, log_probs, values, done):
        """
        Feed transition to memory
        """
        self.memory.save(state, action, reward, log_probs, values, done)

    def step(self, state, action):
        """
        Get log prob and entropy of a given state and action
        :param state:
        :param action:
        :return: action, log prob of action and entropy
        """
        states = [state['obs'] for state in state]
        probs = self.actor_critic.predict("action", self.sess, states)
        legal_actions = [state['legal_actions'] for state in state]
        entropy_ = entropy(probs, axis=-1)
        return action, np.log(probs[0][action]), entropy_

    def eval_step(self, state, action=None):
        """
        Predict action and values with the network
        :param state:
        :param action:
        :return: sampled action, log prob of action, value of state and entropy
        """
        probs = self.actor_critic.predict("action", self.sess, state)
        values = self.actor_critic.predict("values", self.sess, state)[0][0]
        new_probs = remove_illegal(probs[0], state['legal_actions'])
        # Make sure only legal actions can be chosen
        action = np.random.choice(np.arange(len(new_probs)), p=new_probs)
        entropy_ = entropy(probs, axis=-1)
        return action, np.log(probs[0][action]), values, entropy_

    def predict(self, state):
        A = self.predict(state['obs'])
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
        return action

    def train(self):
        loss = 0
        critic_loss = 0
        actor_loss = 0

        for mini_batch_idx in self.memory.get_minibatch_idxs():
            state_batch, action_batch, reward_batch, log_prob_batch, value_batch, done_batch, advantage_batch, returns_batch = self.memory.sample_batch(mini_batch_idx)
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
            old_log_prob = log_prob_batch
            action, new_log_prob, entropy = self.step(state_batch, action_batch)
            ratio = np.exp(new_log_prob - old_log_prob)

            loss, critic_loss, actor_loss = self.actor_critic.update(self.sess, state_batch, ratio, advantage_batch,
                                                                     returns_batch, entropy)
            self.train_t += 1

        return loss, critic_loss, actor_loss


class Estimator:
    """
    Actor-Critic network.
    This network is used for both the Policy and the Value estimations.
    """

    def __init__(self, scope="estimator", action_num=2, value_num=1, learning_rate=0.001, state_shape=None,
                 mlp_layers=None,
                 output_func=None, entropy_beta=0.001, clip_param=0.2):
        """ Initilalize an Estimator object.
        """
        self.scope = scope
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.state_shape = state_shape if isinstance(state_shape, list) else [state_shape]
        self.mlp_layers = map(int, mlp_layers)
        self.output_func = output_func
        self.entropy_beta = entropy_beta
        self.clip_param = clip_param
        self.value_num = value_num

        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam')

        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def _build_model(self):
        """
        Build an MLP model.
        """
        input_shape = [None]
        input_shape.extend(self.state_shape)
        self.X_pl = tf.placeholder(shape=input_shape, dtype=tf.float32, name="X")
        self.values_target_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="values_target")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.ratio = tf.placeholder(shape=[None], dtype=tf.float32, name="ratio")
        self.entropy = tf.placeholder(shape=[None], dtype=tf.float32, name="entropy")
        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name="advantage")

        # Batch Normalization
        # X = tf.layers.batch_normalization(self.X_pl, training=self.is_train)

        # Fully connected layers
        fc = tf.contrib.layers.flatten(self.X_pl)
        for dim in self.mlp_layers:
            fc = tf.contrib.layers.fully_connected(fc, dim, activation_fn=tf.nn.relu)

        # Readout Layers
        self.action_predictions = tf.contrib.layers.fully_connected(fc, self.action_num, activation_fn=self.output_func)
        self.value_predictions = tf.contrib.layers.fully_connected(fc, self.value_num, activation_fn=None)
        # Get the predictions for the chosen actions only
        # gather_indices = tf.range(batch_size) * tf.shape(self.action_predictions)[1] + self.actions_pl
        # self.action_predictions = tf.gather(tf.reshape(self.action_predictions, [-1]), gather_indices)

        # Actor Loss
        surrogate_1 = self.ratio * self.advantage
        surrogate_2 = tf.clip_by_value(self.ratio, 1 - self.clip_param, 1 + self.clip_param) * self.advantage
        self.actor_loss = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2), 0)

        # Critic Loss
        self.critic_loss = 0.5 * tf.losses.mean_squared_error(self.values_target_pl, tf.squeeze(self.value_predictions))
        self.entropy_loss = tf.reduce_mean(self.entropy, 0)
        self.loss = self.actor_loss + self.critic_loss - self.entropy_beta * self.entropy_loss

    def predict(self, pred, sess, state):
        """
        Predicts action and values, depending on input.
        :param pred: str, which prediction to get from the network
        :param sess: the tensorflow Session object
        :param state: the State for which to get the predictions
        :return: either value or action preductions

        """
        if type(state) == dict:
            state = np.expand_dims(state['obs'], 0)
        else:
            state = state

        if pred == "action":
            return sess.run(self.action_predictions, {self.X_pl: state, self.is_train: False})
        elif pred == "values":
            return sess.run(self.value_predictions, {self.X_pl: state, self.is_train: False})

    def update(self, sess, states, ratios, advantages, returns, entropy):
        """

        :param sess: Session
        :param states: State batch
        :param ratios: Ratio batch
        :param advantages: advantage batch
        :param returns: (discounted) returns batch
        :param entropy: entropy batch
        :return: loss
        """
        states = [state['obs'] for state in states]
        feed_dict = {
            self.X_pl: states,
            self.ratio: ratios,
            self.advantage: advantages,
            self.entropy: entropy,
            self.values_target_pl: returns,
            self.is_train: True
        }
        _, _, loss, critic_loss, actor_loss = sess.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.loss, self.critic_loss, self.actor_loss],
            feed_dict)
        return loss, critic_loss, actor_loss


class Memory(object):
    """
    Memory buffer to store transitions, calculate advantages and enabling to sample from buffer.
    """

    def __init__(self, memory_size, batch_size):
        """ Initialize
        Args:
            memory_size (int): the size of the memroy buffer
            batch_size (int): batch size for minibatch sampling
        """

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.rewards = []
        self.values = []
        self.discounted_rewards = []
        self.advantages = []

    def save(self, state, action, reward, log_probs, values, done):
        """
        Save transition to memeory
        :param state:
        :param action:
        :param reward:
        :param log_probs:
        :param values:
        :param done:
        """
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, log_probs, values, done, 0, 0)
        self.memory.append(transition)

    def reset(self):
        """
        Reset the memory
        """
        self.memory = []
        self.rewards = []
        self.values = []
        self.discounted_rewards = []
        self.advantages = []
        print("----- Memory reset --------")

    def calculate_discounted_rewards(self, gamma=0.95):
        """
        Method for calculating discounted rewards
        :param gamma: reward discount factor
        """
        rewards = self.rewards
        self.rewards = []

        G = np.zeros_like(rewards, np.float64)
        temp = 0
        for t in reversed(range(len(rewards))):
            temp = temp * gamma + rewards[t]
            G[t] = temp
        self.discounted_rewards = G

    def calculate_advantages(self):
        """
        Calculating advantages
        :return: advantages
        """
        self.calculate_discounted_rewards()

        for i in range(len(self.discounted_rewards)):
            self.advantages.append(self.discounted_rewards[i] - self.values[i])

        return self.advantages

    def calculate_advantage_gae(self, last_value, next_done, gamma=0.95, gae_lambda=0.95):
        """
        Methof for calculating global advantage estimates and discounted returns
        :param last_value: last value prediction needed for correct calculation
        :param next_done: indication of next value of last state is done
        :param gamma: discount factor
        :param gae_lambda: advantage lambda
        """
        gae = 0
        rewards = [traj.reward for traj in self.memory]
        values = [traj.value for traj in self.memory]
        dones = [traj.done for traj in self.memory]
        ga_estimates = []
        disc_returns = []

        # Calculate GAEs
        for i in reversed(range(len(rewards))):
            if i == self.memory_size - 1:
                next_non_terminal = 1 - next_done
                next_value = last_value
            else:
                next_non_terminal = 1 - dones[i]
                next_value = values[i + 1]

            delta = rewards[i] + gamma * next_value * next_non_terminal - values[i]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            ga_estimates.insert(0, gae)

        # Calculate discounted returns
        for i in range(len(ga_estimates)):
            disc_returns.append(ga_estimates[i] + values[i])

        # Add GAE to trajectories
        for idx, traj in enumerate(self.memory):
            traj.advantage = ga_estimates[idx]
            traj.discounted_reward = disc_returns[idx]

    def sample(self):
        """
        Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            log_probs (list): a batch of log probs
            values (list): a batch of values
            done_batch (list): a batch of dones
            advantage_batch (list): a batch of advantage estimates
        """
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

    def sample_batch(self, batch_idx):
        samples = (np.asarray(self.memory)[batch_idx])
        return map(np.asarray, zip(*samples))

    def get_minibatch_idxs(self):
        idxs = np.arange(self.memory_size)
        np.random.shuffle(idxs)

        return [idxs[start:start + self.batch_size] for start in np.arange(0, self.memory_size, self.batch_size)]

    def is_full(self):
        return len(self.memory) >= self.memory_size
