import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, RNN, GRUCell, Maximum


class QMixer(Model):

    def __init__(self, n_agents, mixing_embed_dim=32, rnn_hidden_dim=64, n_actions=5):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim

        # Agent Network
        self.fc1 = Dense(rnn_hidden_dim, activation=tf.nn.relu)
        # self.rnn = RNN(GRUCell(rnn_hidden_dim))
        self.fc2 = Dense(n_actions, activation=tf.nn.softmax)

        # Hypernetworks
        self.hyper_w_1 = Dense(self.embed_dim * self.n_agents)
        self.hyper_w_final = Dense(self.embed_dim)
        self.hyper_b_1 = Dense(self.embed_dim)
        self.V1 = Dense(self.embed_dim, activation=tf.nn.relu)
        self.V2 = Dense(1)

        # Mixer Network.
        self.action_selection = Maximum()
        self.fc1_mix = Dense(self.embed_dim, activation=tf.nn.relu, trainable=False)
        self.fc2_mix = Dense(1, trainable=False)

        self.state = None

    def call(self, agents_obs, training=False, mask=None):
        x = self.fc1(agents_obs)
        # x = self.rnn(x)
        agents_qs = self.fc2(x)

        if training:
            # Predict weights for the outer model.
            w_1 = tf.abs(self.hyper_w_1(self.state))
            b_1 = self.hyper_b_1(self.state)
            w_f = tf.abs(self.hyper_w_final(self.state))
            v = self.V1(self.state)
            v = self.V2(v)

            # Set the weight predictions as the weight variables on the outer model.
            self.fc1_mix.kernel = tf.reshape(w_1, (self.n_agents, self.embed_dim))
            self.fc1_mix.bias = tf.reshape(b_1, (self.embed_dim,))
            self.fc2_mix.kernel = tf.reshape(w_f, (self.embed_dim, 1))
            self.fc2_mix.bias = tf.reshape(v, (1,))

            # agents_best_qs = tf.expand_dims(tf.math.reduce_max(agents_qs, axis=1), 0)
            inputs = [tf.reshape(agents_qs[:, i], (1, agents_qs.shape[0])) for i in range(agents_qs.shape[1])]
            agents_best_qs = self.action_selection(inputs)
            # Inference on the outer model.
            y = self.fc1_mix(agents_best_qs)
            q_tot = self.fc2_mix(y)

            return q_tot
        else:
            return agents_qs

    def update_state(self, state):
        self.state = state

    def get_config(self):
        return {
            'name': 'rnn_agent_q_mix',
            'layers': [
                # Agent Network
                self.fc1.get_config(),
                # self.rnn.get_config(),
                self.fc2.get_config(),

                # Hypernetworks
                self.hyper_w_1.get_config(),
                self.hyper_w_final.get_config(),
                self.hyper_b_1.get_config(),
                self.V1.get_config(),
                self.V2.get_config(),

                # Mixer Network.
                # self.fc1_mix.get_config(),
                # self.fc2_mix.get_config()
            ]
        }
