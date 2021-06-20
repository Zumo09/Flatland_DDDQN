import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class QMixer(Model):

    def __init__(self, state_shape, n_agents, mixing_embed_dim, optimizer=None):
        super(QMixer, self).__init__()
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


if __name__ == '__main__':
    state_shape_ = (10, 2, 2)
    n_agents_ = 10
    mixing_embed_dim_ = 32

    mixer = QMixer(state_shape_, n_agents_, mixing_embed_dim_)

    steps = 100

    agent_qs_ = np.random.rand(1, n_agents_)
    states_ = np.random.randint(0, 10, state_shape_)
    y_ = np.random.rand(1)

    in_ = (agent_qs_, states_)

    hist = mixer.fit(in_, y_, epochs=steps)

    print()
    print("Ground truth")
    print(y_)
    print("Prediction")
    print(mixer([agent_qs_, states_]))

    import matplotlib.pyplot as plt

    plt.plot(range(steps), hist.history['loss'])
    plt.show()
