from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Flatten
from tensorflow import keras
import tensorflow as tf
import numpy as np


class BaseModel():
    """
    Base class used to implement a model.
    This class itself should not instantiated.
    """

    def __init__(self, comm, controller, rank, n_acts, obs_shape, sess_config=None):
        if sess_config is None:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=sess_config)

        ### Initialize MPI variables ###
        self.comm = comm
        self.controller = controller
        self.rank = rank

        ### Initialize placeholder and other network ops ###
        self.sess_config = sess_config
        self.obs_shape = obs_shape
        self.n_acts = n_acts
        self.update_policy = self.train_policy
        self.create_phs(obs_shape=self.obs_shape)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.create_policy_ops()
        self.sess.run(tf.global_variables_initializer())
        # Sync the weights of models on all processes
        self.sync_weights()

    def create_phs(self, obs_shape):
        """
        Creates placeholders (input ops) for the model.
        """
        self.obs_ph = tf.placeholder(
            dtype=tf.float32, shape=(None, *list(obs_shape)))
        self.next_obs_ph = tf.placeholder(
            dtype=tf.float32, shape=(None, *list(obs_shape)))
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.gae_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

    def create_policy_ops(self):
        """
        Creates the ops for the policy and value network.
        Additionally creates the ops for updating the network.
        This function should be overridden by subclasses.
        """
        self.act_out = 0
        self.value_op = 0
        self.policy_update = 0
        self.value_update = 0

    def gen_actions(self, obs):
        """
        Generates actions (1 to 1) for each observation passed
        into the function.
        """
        if type(obs) == list:
            obs = np.asarray(obs)

        assert (type(obs) == np.ndarray), \
            "Observations must be a numpy array!"
        assert (list(obs.shape)[1:] == list(self.obs_shape)), \
            "Observations must have the shape, (batch_size, dimensions..., 1)!"

        return self.sess.run(self.act_out, feed_dict={self.obs_ph: obs})[0]

    def gen_actions_and_values(self, obs):
        """
        Generates actions and values (1 to 1) for each observation
        passed into the function.
        """
        if type(obs) == list:
            obs = np.asarray(obs)

        assert (type(obs) == np.ndarray), \
            "Observations must be a numpy array!"
        assert (list(obs.shape)[1:] == list(self.obs_shape)), \
            "Observations must have the shape, (batch_size, dimensions..., 1)!"

        acts, vals = self.sess.run(
            [self.act_out, self.value_op], feed_dict={self.obs_ph: obs})
        return acts[0], vals[0]

    def train_policy(self, states, actions, rewards, gaes):
        """
        Updates the policy given training data from
        environment simulation.
        """
        self.sess.run([self.policy_update, self.value_update],
                      feed_dict={self.obs_ph: states,
                                 self.act_ph: actions,
                                 self.rew_ph: rewards,
                                 self.gae_ph: gaes})

    def sync_weights(self):
        """
        Sync the weights between model on all processes
        using MPI.
        """
        if self.rank == self.controller:
            self.comm.bcast(self.sess.run(
                tf.trainable_variables()), self.controller)
        else:
            sync_vars = self.comm.bcast(None, self.controller)
            t_vars = tf.trainable_variables()
            for pair in zip(t_vars, sync_vars):
                self.sess.run(tf.assign(pair[0], pair[1]))


class OneDimModel(BaseModel):
    """
    Vanilla Policy Gradient implemented for an
    environment with 1-dimensional states.
    """

    def __init__(self, comm, controller, rank, n_acts, obs_shape, sess_config=None):
        super().__init__(comm, controller, rank, n_acts, obs_shape, sess_config)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    def create_policy_ops(self):
        """
        Creates the ops for the policy and value network.
        Additionally creates the ops for updating the network.
        """
        with tf.variable_scope('policy'):
            # Creating a fully connected net for the policy and value estimator
            dense_1 = Dense(64, activation='relu')(self.obs_ph)

            act_dense_1 = Dense(64, activation='relu')(dense_1)
            act_dense_2 = Dense(64, activation='relu')(act_dense_1)

            val_dense_1 = Dense(64, activation='relu')(dense_1)
            val_dense_2 = Dense(64, activation='relu')(val_dense_1)

            # Output probability distribution over possible actions
            self.act_probs_op = Dense(
                self.n_acts, activation='softmax', name='act_probs')(act_dense_2)
            self.act_out = tf.random.categorical(tf.log(self.act_probs_op), 1)

            # Output value of observed state
            self.value_op = Dense(1, activation='linear')(val_dense_2)

            self.act_masks = tf.one_hot(
                self.act_ph, self.n_acts, dtype=tf.float32)
            self.log_probs = tf.log(self.act_probs_op)

            self.resp_acts = tf.reduce_sum(
                self.act_masks * self.log_probs, axis=1)
            self.policy_loss = \
                -tf.reduce_mean(self.resp_acts * self.gae_ph)

            self.policy_update = self.optimizer.minimize(self.policy_loss)

            # Add dependency on policy update to make sure the value network
            # only gets updated after the policy
            with tf.control_dependencies([self.policy_update]):
                self.value_loss = tf.reduce_mean(
                    tf.square(self.rew_ph - tf.squeeze(self.value_op)))
                self.value_update = self.optimizer.minimize(self.value_loss)


class TwoDimModel(BaseModel):
    """
    Vanilla Policy Gradient implemented for an
    environment with 2-dimensional states.
    """

    def __init__(self, comm, controller, rank, n_acts, obs_shape, sess_config=None):
        super().__init__(comm, controller, rank, n_acts, obs_shape, sess_config)

    def create_policy_ops(self):
        """
        Creates the ops for the policy and value network.
        Additionally creates the ops for updating the network.
        """
        with tf.variable_scope('policy'):
            # Creating a conv net for the policy and value estimator
            conv_1 = Conv2D(16, 5, 3, activation='relu')(self.obs_ph)
            pooling_1 = MaxPool2D(2)(conv_1)

            act_conv_1 = Conv2D(32, 3, 2, activation='relu')(conv_1)
            act_pool_1 = MaxPool2D(2)(act_conv_1)
            act_flat = Flatten()(act_pool_1)

            val_conv_1 = Conv2D(32, 3, 2, activation='relu')(conv_1)
            val_pool_1 = MaxPool2D(2)(val_conv_1)
            val_flat = Flatten()(val_pool_1)

            # Output probability distribution over possible actions
            self.act_probs_op = Dense(
                self.n_acts, activation='softmax', name='act_probs')(act_flat)
            self.act_out = tf.random.categorical(tf.log(self.act_probs_op), 1)

            # Output value of observed state
            self.value_op = Dense(1, activation='linear')(val_flat)

            self.act_masks = tf.one_hot(
                self.act_ph, self.n_acts, dtype=tf.float32)
            self.log_probs = tf.log(self.act_probs_op)

            self.resp_acts = tf.reduce_sum(
                self.act_masks * self.log_probs, axis=1)
            self.policy_loss = \
                -tf.reduce_mean(self.resp_acts * self.gae_ph)

            self.policy_update = self.optimizer.minimize(self.policy_loss)

            # Add dependency on policy update to make sure the value network
            # only gets updated after the policy
            with tf.control_dependencies([self.policy_update]):
                self.value_loss = tf.reduce_mean(
                    tf.square(self.rew_ph - tf.squeeze(self.value_op)))
                self.value_update = self.optimizer.minimize(self.value_loss)
