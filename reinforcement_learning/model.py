import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import mean
from tensorflow.keras.models import load_model


def DuelingQNetwork(state_size, action_size, learning_rate=1e-4, hidsize1=256, hidsize2=128):
    observations = Input(shape=(state_size,), name='observation')
    x = Dense(hidsize1, activation="relu", name='dense_1')(observations)

    val = Dense(hidsize1, activation='relu', name='val_1')(x)
    val = Dense(hidsize2, activation='relu', name='val_2')(val)
    val = Dense(1, name='val_3')(val)

    # advantage calculation
    adv = Dense(hidsize1, activation='relu', name='adv_1')(x)
    adv = Dense(hidsize2, activation='relu', name='adv_2')(adv)
    adv = Dense(action_size, name='adv_3')(adv)

    q_values = val + adv - mean(adv, axis=1, keepdims=True)

    model = Model(inputs=observations, outputs=q_values)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())

    return model


def load_dueling_dqn(path):
    return load_model(path)
