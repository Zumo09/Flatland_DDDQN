import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import mean
from tensorflow.keras.models import load_model


def dueling_dqn(observation_shape, action_size, learning_rate=1e-4, hidden=1024, hidden_1=256, hidden_2=128):
    observations = Input(shape=observation_shape, name='observation')
    x = Dense(hidden, activation="relu", name='dense_1')(observations)

    val = Dense(hidden_1, activation='relu', name='val_1')(x)
    val = Dense(hidden_2, activation='relu', name='val_2')(val)
    val = Dense(1, name='val_3')(val)

    # advantage calculation
    adv = Dense(hidden_1, activation='relu', name='adv_1')(x)
    adv = Dense(hidden_2, activation='relu', name='adv_2')(adv)
    adv = Dense(action_size, name='adv_3')(adv)

    q_values = val + adv - mean(adv, axis=1, keepdims=True)

    model = Model(inputs=observations, outputs=q_values)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())

    return model


def load_dueling_dqn(path):
    return load_model(path)
