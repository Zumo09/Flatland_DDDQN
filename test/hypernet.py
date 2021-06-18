import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


class Mixer:

    def __init__(self):
        self.input_dim = 784
        self.classes = 10

        # This is the model we'll actually use to predict labels (the hypernetwork).
        self.outer_model = Sequential(
            [Dense(64, activation=tf.nn.relu), Dense(self.classes), ]
        )

        # It doesn't need to create its own weights, so let's mark its layers
        # as already built. That way, calling `outer_model` won't create new variables.
        for layer in self.outer_model.layers:
            layer.built = True

        # This is the number of weight coefficients to generate. Each layer in the
        # hypernetwork requires output_dim * input_dim + output_dim coefficients.
        self.num_weights_to_generate = (self.classes * 64 + self.classes) + (64 * self.input_dim + 64)

        # This is the model that generates the weights of the `outer_model` above.
        self.inner_model = Sequential(
            [
                Dense(16, activation=tf.nn.relu),
                Dense(self.num_weights_to_generate, activation=tf.nn.sigmoid),
            ]
        )

        # Loss and optimizer.
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # Predict weights for the outer model.
            weights_pred = self.inner_model(x)

            # Reshape them to the expected shapes for w and b for the outer model.
            # Layer 0 kernel.
            start_index = 0
            w0_shape = (self.input_dim, 64)
            w0_coeffs = weights_pred[:, start_index: start_index + np.prod(w0_shape)]
            w0 = tf.reshape(w0_coeffs, w0_shape)
            start_index += np.prod(w0_shape)
            # Layer 0 bias.
            b0_shape = (64,)
            b0_coeffs = weights_pred[:, start_index: start_index + np.prod(b0_shape)]
            b0 = tf.reshape(b0_coeffs, b0_shape)
            start_index += np.prod(b0_shape)
            # Layer 1 kernel.
            w1_shape = (64, self.classes)
            w1_coeffs = weights_pred[:, start_index: start_index + np.prod(w1_shape)]
            w1 = tf.reshape(w1_coeffs, w1_shape)
            start_index += np.prod(w1_shape)
            # Layer 1 bias.
            b1_shape = (self.classes,)
            b1_coeffs = weights_pred[:, start_index: start_index + np.prod(b1_shape)]
            b1 = tf.reshape(b1_coeffs, b1_shape)
            start_index += np.prod(b1_shape)

            # Set the weight predictions as the weight variables on the outer model.
            self.outer_model.layers[0].kernel = w0
            self.outer_model.layers[0].bias = b0
            self.outer_model.layers[1].kernel = w1
            self.outer_model.layers[1].bias = b1

            # Inference on the outer model.
            preds = self.outer_model(x)
            loss = self.loss_fn(y, preds)

        # Train only inner model.
        grads = tape.gradient(loss, self.inner_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.inner_model.trainable_weights))
        return loss

    def predict(self, x):
        return tf.argmax(self.outer_model(x))


if __name__ == '__main__':
    # Prepare a dataset.
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
    )

    # We'll use a batch size of 1 for this experiment.
    dataset = dataset.shuffle(buffer_size=1024).batch(1)

    model = Mixer()

    losses = []  # Keep track of the losses over time.
    for step, (x_, y_) in enumerate(dataset):
        loss_ = model.train_step(x_, y_)

        # Logging.
        losses.append(float(loss_))
        if step % 100 == 0:
            print("Step:", step, "Loss:", sum(losses) / len(losses))

        # Stop after 1000 steps.
        # Training the model to convergence is left
        # as an exercise to the reader.
        if step >= 1000:
            break

    import matplotlib.pyplot as plt

    plt.plot(range(len(losses)), losses)
    plt.show()

    model.inner_model.summary()
    model.outer_model.summary()
