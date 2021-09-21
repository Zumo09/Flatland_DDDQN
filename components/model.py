from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Add, Average, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import mean
from tensorflow.keras.models import load_model


def dueling_q_network(state_size, action_size, params):
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


def _preprocessing_network(inputs, state_size, params):
    x = Dense(params.hidden_size_1, activation="relu", name='dense_0')(inputs)

    return x


def _dueling_network(inputs, action_size, params, index=0):
    # value calculation
    val = Dense(params.hidden_size_2, activation='relu', name='val_0_' + str(index))(inputs)
    val = Dense(params.hidden_size_3, activation='relu', name='val_1_' + str(index))(val)
    val = Dense(1, name='val_2_' + str(index))(val)

    # advantage calculation
    adv = Dense(params.hidden_size_2, activation='relu', name='adv_0_' + str(index))(inputs)
    adv = Dense(params.hidden_size_3, activation='relu', name='adv_1_' + str(index))(adv)
    adv = Dense(action_size, name='adv_2_' + str(index))(adv)

    # q value calculation
    add = Add(name='add_' + str(index))([val, adv])
    average = mean(adv, axis=1, keepdims=True)
    q_values = Subtract(name='q_values_' + str(index))([add, average])

    return q_values


def bootstrapped_dueling_q_network(state_size, action_size, params):
    inputs = Input(shape=(state_size,), name='observation')

    prep_net = _preprocessing_network(inputs, state_size, params)

    heads = []
    for i in range(params.num_heads):
        head = _dueling_network(prep_net, action_size, params, i)
        heads.append(head)

    model = Model(inputs=inputs, outputs=heads, name='bootstrapped_dueling_q_network_' + str(params.num_heads) + '_heads')

    model.compile(optimizer=Adam(learning_rate=params.learning_rate), loss=MeanSquaredError())

    return model
