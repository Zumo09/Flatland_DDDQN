import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, RNN, GRUCell


class RnnAgent(Model):
    def __init__(self, rnn_hidden_dim=64, n_actions=5):
        super(RnnAgent, self).__init__()

        self.fc1 = Dense(rnn_hidden_dim, activation=tf.nn.relu)
        self.rnn = RNN(GRUCell(rnn_hidden_dim))
        self.fc2 = Dense(n_actions, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.rnn(x)
        return self.fc2(x)

    def get_config(self):
        return {
            'name': 'rnn_agent',
            'layers': [
                self.fc1.get_config(),
                self.rnn.get_config(),
                self.fc2.get_config()
            ]
        }


class DDQNAgent(Model):
    def __init__(self, action_size, hidden_1=128, hidden_2=128):
        super(DDQNAgent, self).__init__()

        self.fc1_val = Dense(hidden_1, activation=tf.nn.relu)
        self.fc2_val = Dense(hidden_2, activation=tf.nn.relu)
        self.fc3_val = Dense(1)

        self.fc1_adv = Dense(hidden_1, activation=tf.nn.relu)
        self.fc2_adv = Dense(hidden_2, activation=tf.nn.relu)
        self.fc3_adv = Dense(action_size)

    def call(self, inputs, training=None, mask=None):
        val = self.fc1_val(inputs)
        val = self.fc2_val(val)
        val = self.fc3_val(val)

        # advantage calculation
        adv = self.fc1_adv(inputs)
        adv = self.fc2_adv(adv)
        adv = self.fc3_adv(adv)
        return val + adv - tf.reduce_mean(adv)

    def get_config(self):
        return {
            'name': 'QNetwork',
            'layers': [
                self.fc1_val.get_config(),
                self.fc2_val.get_config(),
                self.fc3_val.get_config(),
                self.fc1_adv.get_config(),
                self.fc2_adv.get_config(),
                self.fc3_adv.get_config()
            ]
        }


class Mixer(Model):

    def __init__(self, state_shape, n_agents, mixing_embed_dim, optimizer=None):
        super(Mixer, self).__init__()
        self.state_dim = np.prod(state_shape)
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim

        # Hypernetworks
        self.hyper_w_1 = Dense(self.embed_dim * self.n_agents, input_shape=(self.state_dim,))

        self.hyper_w_final = Dense(self.embed_dim, input_shape=(self.state_dim,))

        self.hyper_b_1 = Dense(self.embed_dim, input_shape=(self.state_dim,))

        self.V1 = Dense(self.embed_dim, activation=tf.nn.relu, input_shape=(self.state_dim,))
        self.V2 = Dense(1)

        # This is the model we'll actually use.
        self.fc1 = Dense(self.embed_dim, activation=tf.nn.relu, input_shape=(n_agents,), trainable=False)
        self.fc1 = Dense(1, trainable=False)

        # Loss and optimizer.
        loss_fn = tf.keras.losses.MeanSquaredError()

        if optimizer is None:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-4)

        self.compile(optimizer=optimizer, loss=loss_fn)

    def call(self, inputs, training=None, mask=None):
        agent_qs, states = inputs

        states = states.reshape((1, self.state_dim))
        # Predict weights for the outer model.
        w_1 = tf.abs(self.hyper_w_1(states))
        b_1 = self.hyper_b_1(states)
        w_f = tf.abs(self.hyper_w_final(states))
        v = self.V1(states)
        v = self.V2(v)

        # Set the weight predictions as the weight variables on the outer model.
        self.fc1.kernel = tf.reshape(w_1, (self.n_agents, self.embed_dim))
        self.fc1.bias = tf.reshape(b_1, self.embed_dim)
        self.fc2.kernel = tf.reshape(w_f, (self.embed_dim, 1))
        self.fc2.bias = tf.reshape(v, 1)

        # Inference on the outer model.
        q_tot = self.fc1(agent_qs)
        return self.fc2(q_tot)

    def get_config(self):
        pass
