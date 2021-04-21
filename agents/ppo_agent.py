"""
Implementation of our PPO Agent
"""
import tensorflow as tf
from agents.dou_env import *
import random
from dataclasses import dataclass, astuple



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

        :param ts: Tuple of transitions
        """
        (state, action_info, reward, next_state, done) = tuple(ts)

        action, log_prob, value, entropy = action_info

        self.feed_memory(state, action, reward, log_prob, value, done)

        # TODO: Train. When to train ?

        # self.total_t += 1
        # if self.total_t %  self.memory.memory_size == 0 :
        #     print("Memory full, now training")
        #     last_value = self.actor_critic.predict("values", self.sess, state)[0][0]
        #     self.memory.calculate_advantage_gae(last_value)
        #     self.train()
        #     self.memory.reset()

    def feed_memory(self, state, action, reward, log_probs, values, done):
        ''' Feed transition to memory

        '''
        self.memory.save(state, action, reward, log_probs, values, done)

    def step(self, state, action):
        states = [state['obs'] for state in state]
        probs = self.actor_critic.predict("action", self.sess, states)
        legal_actions = [state['legal_actions'] for state in state]
        new_probs = remove_illegal(probs[0], legal_actions[0])
        #dist = tf.distributions.Categorical(probs=new_probs)

        entropy = -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(new_probs), new_probs)).eval()

        # return action, dist.log_prob(action).eval(), entropy  # dist.entropy().eval()
        return action, np.log(new_probs[action]), entropy  # dist.entropy().eval()

    def eval_step(self, state, action=None):

        probs = self.actor_critic.predict("action", self.sess, state)
        values = self.actor_critic.predict("values", self.sess, state)[0][0]
        new_probs = remove_illegal(probs[0], state['legal_actions'])
        #print(new_probs)
        #dist = tf.distributions.Categorical(probs=new_probs, allow_nan_stats=False)
        # if action is None:
        #action = dist.sample()
        action = np.random.choice(np.arange(len(new_probs)), p=new_probs)
        #print("Action: ", action)
        # A = np.ones(self.action_num, dtype=float)
        # A = remove_illegal(A, state['legal_actions'])
        # action = np.random.choice(np.arange(len(A)), p=A)

        #entropy = -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(new_probs), new_probs)).eval()
        entropy = - np.sum(np.log(probs)*probs)
        return action, np.log(new_probs[action]), values, entropy# entropy #dist.log_prob(action).eval(), 0,0 #values,  0 #entropy  # dist.entropy().eval()

    def predict(self, state):
        A = self.predict(state['obs'])
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
        return action

    def train(self):

        #for t in range(self.training_steps):
        #print(f"Train Step: {self.train_t}")
        state_batch, action_batch, reward_batch, log_prob_batch, value_batch, done_batch, advantage_batch, returns_batch = self.memory.sample()
        old_log_prob = log_prob_batch
        action, new_log_prob, entropy = self.step(state_batch, action_batch)
        ratio = np.exp(new_log_prob - old_log_prob)

        # # Test prints
        # print("Ratio:", ratio)
        # print("Advantage Batch", advantage_batch)
        # print("Value Batch:", value_batch)
        # print("New log prob:", new_log_prob)
        # print("Entropy:", entropy)

        #print("Updating...")
        loss = self.actor_critic.update(self.sess, state_batch, ratio, advantage_batch, returns_batch, entropy)
        #print("Loss:", loss)
        self.train_t += 1


class Estimator:
    """
    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", action_num=2, value_num=1, learning_rate=0.001, state_shape=None,
                 mlp_layers=None,
                 output_func=None, entropy_beta=0.001, clip_param=0.2):
        """ Initilalize an Estimator object.

        Args:
            action_num (int): the number output actions
            state_shap (list): the shape of the state space
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
        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam')

        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def _build_model(self):
        """
        Build an MLP model.
        """
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        input_shape = [None]
        input_shape.extend(self.state_shape)
        self.X_pl = tf.placeholder(shape=input_shape, dtype=tf.float32, name="X")
        # The TD target value
        self.values_target_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="values_target")
        #self.values_preds_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="values_preds")
        #self.values_target_pl = tf.expand_dims(self.values_target_pl, axis=1)
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        # Boolean to indicate whether is training or not
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.ratio = tf.placeholder(shape=[None], dtype=tf.float32, name="ratio")
        self.entropy = tf.placeholder(dtype=tf.float32, name="entropy")
        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name="advantage")

        batch_size = tf.shape(self.X_pl)[0]

        # Batch Normalization
        X = tf.layers.batch_normalization(self.X_pl, training=self.is_train)

        # Fully connected layers
        fc = tf.contrib.layers.flatten(X)
        for dim in self.mlp_layers:
            fc = tf.contrib.layers.fully_connected(fc, dim, activation_fn=tf.tanh)
        # Readout Layers
        self.action_predictions = tf.contrib.layers.fully_connected(fc, self.action_num, activation_fn=self.output_func)
        print(f"Action Predictions: {self.action_predictions}")
        self.value_predictions = tf.contrib.layers.fully_connected(fc, self.value_num, activation_fn=None)
        print(f"Value Predictions: {self.value_predictions}")
        # Get the predictions for the chosen actions only
        # gather_indices = tf.range(batch_size) * tf.shape(self.action_predictions)[1] + self.actions_pl
        # self.action_predictions = tf.gather(tf.reshape(self.action_predictions, [-1]), gather_indices)

        # Actor Loss
        surrogate_1 = self.ratio * self.advantage
        surrogate_2 = tf.clip_by_value(self.ratio, 1 - self.clip_param, 1 + self.clip_param) * self.advantage
        actor_loss = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2), 0)

        # Critic Loss
        critic_loss = 0.5 * tf.losses.mean_squared_error(self.values_target_pl, self.value_predictions[0])
        self.loss = actor_loss + critic_loss - self.entropy_beta * self.entropy

    def predict(self, pred, sess, state):
        """ Predicts action values.

        Args:
          sess (tf.Session): Tensorflow Session object
          s (numpy.array): State input of shape [batch_size, 4, 160, 160, 3]
          is_train (boolean): True if is training

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
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
        """ Updates the estimator towards the given targets.

        Args:
          sess (tf.Session): Tensorflow Session object
          s (list): State input of shape [batch_size, 4, 160, 160, 3]
          a (list): Chosen actions of shape [batch_size]
          y (list): Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
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
        _, _, loss = sess.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        return loss


class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.rewards = []
        self.values = []
        self.discounted_rewards = []
        self.advantages = []

    def save(self, state, action, reward, log_probs, values, done):
        """ Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            done (boolean): whether the episode is finished
        """
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, log_probs, values, done, 0, 0)
        self.memory.append(transition)

    def reset(self):
        self.memory = []
        self.rewards = []
        self.values = []
        self.discounted_rewards = []
        self.advantages = []
        print("----- Memory reset --------")

    def calculate_discounted_rewards(self, gamma=0.95):

        rewards = self.rewards
        self.rewards = []

        G = np.zeros_like(rewards, np.float64)
        temp = 0
        for t in reversed(range(len(rewards))):
            temp = temp * gamma + rewards[t]
            G[t] = temp
        self.discounted_rewards = G

    def calculate_advantages(self):

        self.calculate_discounted_rewards()

        for i in range(len(self.discounted_rewards)):
            self.advantages.append(self.discounted_rewards[i] - self.values[i])

        return self.advantages

    def calculate_advantage_gae(self, last_value, gamma=0.95, gae_lambda=0.95):
        gae = 0
        rewards = [traj.reward for traj in self.memory]
        values = [traj.value for traj in self.memory]
        values.append(last_value)
        dones = [traj.done for traj in self.memory]
        ga_estimates = []
        disc_returns = []

        # Calculate GAEs
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            ga_estimates.insert(0, gae)

        # Calculate discounted returns
        for i in range(len(ga_estimates)):
            disc_returns.append(ga_estimates[i] + values[i])

        # Add GAE to trajectories
        for idx, traj in enumerate(self.memory):
            traj.advantage = ga_estimates[idx]
            traj.discounted_reward = disc_returns[idx]

    def sample(self):
        """ Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            log_probs (list): a batch of log probs
            values (list): a batch of values
            done_batch (list): a batch of dones
            advantage_batch (list): a batch of advantage estimates
        """
        # Needed: State, action , log prob, advantage, value
        # Transition has state, action, reward, log_probs, values, done
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

    def is_full(self):
        return len(self.memory) >= self.memory_size



