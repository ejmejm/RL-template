from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Flatten
from tensorflow import keras
import tensorflow as tf
import numpy as np


class Model():
    def __init__(self, comm, controller, rank, n_acts, sess_config=None, obs_shape=(42, 42)):
        if sess_config is None:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=sess_config)

        self.comm = comm
        self.controller = controller
        self.rank = rank
        self.sess_config = sess_config
        self.obs_shape = obs_shape
        self.n_acts = n_acts
        self.update_policy = self.train_policy
        self.create_phs(obs_shape=self.obs_shape)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.create_policy_ops()
        self.sess.run(tf.global_variables_initializer())
        self.sync_weights()

    def create_phs(self, obs_shape=(42, 42)):
        # Placeholders
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
        Additionally creates the  
        """
        with tf.variable_scope('policy'):
            # Creating a conv net for the policy and value estimator
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
            print(self.resp_acts * self.gae_ph)

            self.policy_update = self.optimizer.minimize(self.policy_loss)

            # Add dependency on policy update to make sure the value network
            # only gets updated after the policy
            with tf.control_dependencies([self.policy_update]):
                self.value_loss = tf.reduce_mean(
                    tf.square(self.rew_ph - tf.squeeze(self.value_op)))
                self.value_update = self.optimizer.minimize(self.value_loss)

    def gen_actions(self, obs):
        if type(obs) == list:
            obs = np.asarray(obs)

        assert (type(obs) == np.ndarray), \
            "Observations must be a numpy array!"
        assert (list(obs.shape)[1:] == list(self.obs_shape)), \
            "Observations must have the shape, (batch_size, dimensions..., 1)!"

        return self.sess.run(self.act_out, feed_dict={self.obs_ph: obs})[0]

    def gen_actions_and_values(self, obs):
        if type(obs) == list:
            obs = np.asarray(obs)

        assert (type(obs) == np.ndarray), \
            "Observations must be a numpy array!"
        assert (list(obs.shape)[1:] == list(self.obs_shape)), \
            "Observations must have the shape, (batch_size, dimensions..., 1)!"

        acts, vals = self.sess.run([self.act_out, self.value_op], feed_dict={self.obs_ph: obs})
        return acts[0], vals[0]

    def train_policy(self, states, actions, rewards, gaes):
        self.sess.run([self.policy_update, self.value_update],
                      feed_dict={self.obs_ph: states,
                                 self.act_ph: actions,
                                 self.rew_ph: rewards,
                                 self.gae_ph: gaes})

    def sync_weights(self):
        if self.rank == self.controller:
            self.comm.bcast(self.sess.run(
                tf.trainable_variables()), self.controller)
        else:
            sync_vars = self.comm.bcast(None, self.controller)
            t_vars = tf.trainable_variables()
            for pair in zip(t_vars, sync_vars):
                self.sess.run(tf.assign(pair[0], pair[1]))
