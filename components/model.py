from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import mean
from tensorflow.keras.models import load_model


def DuelingQNetwork(state_size, action_size, params):
    observations = Input(shape=(state_size,), name='observation')
    x = Dense(params.hidden_size_1, activation="relu", name='dense_1')(observations)

    val = Dense(params.hidden_size_2, activation='relu', name='val_1')(x)
    val = Dense(params.hidden_size_3, activation='relu', name='val_2')(val)
    val = Dense(1, name='val_3')(val)

    # advantage calculation
    adv = Dense(params.hidden_size_2, activation='relu', name='adv_1')(x)
    adv = Dense(params.hidden_size_3, activation='relu', name='adv_2')(adv)
    adv = Dense(action_size, name='adv_3')(adv)

    q_values = val + adv - mean(adv, axis=1, keepdims=True)

    model = Model(inputs=observations, outputs=q_values)

    model.compile(optimizer=Adam(learning_rate=params.learning_rate), loss=MeanSquaredError())

    return model


def load_dueling_dqn(path):
    return load_model(path)
