import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, RNN, GRUCell


class RnnAgent:

    def __init__(self, input_shape,
                 rnn_hidden_dim,
                 n_actions,
                 optimizer=None):
        inputs = Input(shape=input_shape)
        x = Dense(rnn_hidden_dim, activation=tf.nn.relu)(inputs)
        x = RNN(GRUCell(rnn_hidden_dim))(x)
        q = Dense(n_actions, activation=tf.nn.softmax)(x)

        self.model = Model(inputs=inputs, outputs=q)

        if optimizer is None:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-4)

        self.model.compile(optimizer=optimizer)

    def __call__(self, inputs):
        return self.model(inputs)

    def apply_gradient(self, loss):
        pass
