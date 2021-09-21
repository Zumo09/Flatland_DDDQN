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
    q_values = val + adv - mean(adv, axis=1, keepdims=True)

    return q_values


def bootstrapped_dueling_q_network(state_size, action_size, params):
    inputs = Input(shape=(state_size,), name='observation')

    prep_net = Dense(params.hidden_size_1, activation="relu", name='dense_0')(inputs)

    heads = []
    for i in range(params.num_heads):
        head_q = _dueling_network(prep_net, action_size, params, i)

        model = Model(inputs=inputs, outputs=head_q, name=f'bootstrapped_ddqn_head_{i}_of_{params.num_heads}')

        model.compile(optimizer=Adam(learning_rate=params.learning_rate), loss=MeanSquaredError())

        heads.append(model)

    return heads
