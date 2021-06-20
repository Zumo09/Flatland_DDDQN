import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, RNN, GRUCell


class RnnAgent(Model):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions, optimizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = Dense(rnn_hidden_dim, activation=tf.nn.relu, input_shape=input_shape)
        # self.rnn = RNN(GRUCell(rnn_hidden_dim))
        self.fc2 = Dense(n_actions, activation=tf.nn.softmax)

        if optimizer is None:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-4)

        self.compile(optimizer=optimizer)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        # x = self.rnn(x)
        return self.fc2(x)

    def get_config(self):
        cfg = {
            'name': 'rnn_agent',
            'layers': [
                self.fc1.get_config(),
                # self.rnn.get_config(),
                self.fc2.get_config()
            ]
        }
        return cfg


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    input_shape_ = (10,)
    rnn_hidden_dim_ = 10
    n_actions_ = 5

    agent = RnnAgent(input_shape_, rnn_hidden_dim_, n_actions_)

    steps = 100
    size = 10

    in_ = np.random.rand(10, *input_shape_)
    y_ = np.random.rand(10, n_actions_)

    print(in_, in_[0].shape)
    print(y_, y_[0].shape)

    print(f'GT: {y_}')

    hist = agent.fit(in_, y_, batch_size=1, epochs=steps)

    plt.plot(range(steps), hist.history['loss'])
    plt.show()

    agent.summary()

