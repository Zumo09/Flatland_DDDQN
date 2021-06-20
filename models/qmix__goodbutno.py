import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class QMixer:

    def __init__(self, state_shape, n_agents, mixing_embed_dim, optimizer=None):
        self.state_dim = np.prod(state_shape)
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim

        # Hypernetworks
        self.hyper_w_1 = Sequential([Dense(self.embed_dim * self.n_agents,
                                           input_shape=(self.state_dim,))],
                                    name='hyper_w_1')
        self.hyper_w_final = Sequential([Dense(self.embed_dim,
                                               input_shape=(self.state_dim,))],
                                        name='hyper_w_final')

        self.hyper_b_1 = Sequential([Dense(self.embed_dim,
                                           input_shape=(self.state_dim,))],
                                    name='hyper_b_1')
        self.V = Sequential([
            Dense(self.embed_dim, activation=tf.nn.relu,
                  input_shape=(self.state_dim,)),
            Dense(1)
        ], name='hyper_V')

        # This is the model we'll actually use.
        self.outer_model = Sequential([
            Dense(self.embed_dim, activation=tf.nn.relu, input_shape=(n_agents,)),
            Dense(1)
        ], name='outer_model')

        # It doesn't need to create its own weights, so let's mark its layers
        # as already built. That way, calling `outer_model` won't create new variables.
        for layer in self.outer_model.layers:
            layer.built = True

        # Loss and optimizer.
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-4)
        else:
            self.optimizer = optimizer

    def train_step(self, agent_qs, states, y):
        with tf.GradientTape() as tape:
            states = states.reshape((1, self.state_dim))
            # Predict weights for the outer model.
            w_1 = tf.abs(self.hyper_w_1(states))
            b_1 = self.hyper_b_1(states)
            w_f = tf.abs(self.hyper_w_final(states))
            v = self.V(states)

            # Set the weight predictions as the weight variables on the outer model.
            self.outer_model.layers[0].kernel = tf.reshape(w_1, (self.n_agents, self.embed_dim))
            self.outer_model.layers[0].bias = tf.reshape(b_1, self.embed_dim)
            self.outer_model.layers[1].kernel = tf.reshape(w_f, (self.embed_dim, 1))
            self.outer_model.layers[1].bias = tf.reshape(v, 1)

            # Inference on the outer model.
            q_tot = self.outer_model(agent_qs)
            loss = self.loss_fn(y, q_tot)

        # Train only inner models.
        grads_w_1, grads_b_1, grads_w_f, grads_V = tape.gradient(loss, [
            self.hyper_w_1.trainable_weights,
            self.hyper_b_1.trainable_weights,
            self.hyper_w_final.trainable_weights,
            self.V.trainable_weights
        ])

        self.optimizer.apply_gradients(zip(grads_w_1, self.hyper_w_1.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_b_1, self.hyper_b_1.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_w_f, self.hyper_w_final.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_V, self.V.trainable_weights))

        return loss


if __name__ == '__main__':
    state_shape_ = (10, 2, 2)
    n_agents_ = 10
    mixing_embed_dim_ = 32

    mixer = QMixer(state_shape_, n_agents_, mixing_embed_dim_)

    steps = 100
    losses = []
    agent_qs_ = np.random.rand(1, n_agents_)
    states_ = np.random.randint(0, 10, state_shape_)
    y_ = np.random.rand(1)

    for i in range(steps):
        loss_ = mixer.train_step(agent_qs_, states_, y_)
        losses.append(loss_)
        print(f'step {i}\t Loss: {loss_}')

    print()
    print("Ground truth")
    print(y_)
    print("Prediction")
    print(mixer.outer_model(agent_qs_))

    import matplotlib.pyplot as plt

    plt.plot(range(steps), losses)
    plt.show()
