import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, RNN, GRUCell


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
        self.fc1_mix = Dense(self.embed_dim, activation=tf.nn.relu, trainable=False)
        self.fc2_mix = Dense(1, trainable=False)

    def call(self, inputs, training=False, mask=None):
        if training:
            states, agents_obs = inputs  # here all the observations refer to the same state
        else:
            agents_obs = inputs

        x = self.fc1(agents_obs)
        # x = self.rnn(x)
        agents_qs = self.fc2(x)

        # TODO: risolvere problema di cambiare le hypernetwork per ogni stato

        if training:
            q_tot = tf.zeros(len(states))
            for i in range(len(states)):
                state = states[i]
                # Predict weights for the outer model.
                w_1 = tf.abs(self.hyper_w_1(state))
                b_1 = self.hyper_b_1(state)
                w_f = tf.abs(self.hyper_w_final(state))
                v = self.V1(state)
                v = self.V2(v)

                # Set the weight predictions as the weight variables on the outer model.
                self.fc1_mix.kernel = tf.reshape(w_1, (self.n_agents, self.embed_dim))
                self.fc1_mix.bias = tf.reshape(b_1, self.embed_dim)
                self.fc2_mix.kernel = tf.reshape(w_f, (self.embed_dim, 1))
                self.fc2_mix.bias = tf.reshape(v, 1)

                agents_best_qs = tf.gather(agents_qs, axis=1,
                                           indices=tf.argmax(agents_qs, axis=1))
                # Inference on the outer model.
                y = self.fc1_mix(agents_best_qs)
                q_tot[i] = self.fc2_mix(y)

            return agents_qs, q_tot
        else:
            return agents_qs

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
