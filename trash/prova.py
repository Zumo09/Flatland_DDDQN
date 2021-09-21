from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np


def _preprocessing_network(inputs):
    x = Dense(8, activation="relu", name='dense_0')(inputs)
    return x


def _dueling_network(inputs):
    # value calculation
    val = Dense(4, activation='relu', name='val_0_')(inputs)
    return val


def bootstrapped_dueling_q_network(state_size, action_size, params):
    inputs = Input(shape=(state_size,), name='observation')

    prep_net = _preprocessing_network(inputs)

    heads = []
    for i in range(params.num_heads):
        head = _dueling_network(prep_net)
        heads.append(head)

    model = Model(inputs=inputs, outputs=heads, name='bootstrapped_dueling_q_network_' + str(params.num_heads) + '_heads')

    model.compile(optimizer=Adam(learning_rate=params.learning_rate), loss=MeanSquaredError())

    return model

if __name__ == '__main__':
    state_size_=4
    states = np.random.random((4, state_size_))
    mod =