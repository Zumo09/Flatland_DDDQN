from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as k

def dueling_dqn(observation_shape, action_size, learning_rate, hidden=1024, hidden_1=256, hidden_2=128):
    observations = Input(shape=(observation_shape,))
    x = Dense(hidden, activation="relu")(observations)

    val = Dense(hidden_1, activation='relu')(x)
    val = Dense(hidden_2, activation='relu')(val)
    val = Dense(1)(val)

    # advantage calculation
    adv = Dense(hidden_1, activation='relu')(x)
    adv = Dense(hidden_2, activation='relu')(adv)
    adv = Dense(action_size)(adv)

    q_values = val + adv - k.mean(adv, axis=1, keepdims=True)))

    model = Model(inputs=observations, outputs=q_values)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())

    return model


class QNetwork(Model):
    def __init__(self, action_size, hidden=1024, hidden_1=256, hidden_2=128):
        super(QNetwork, self).__init__()
        self.fc = Dense(hidden, activation='relu')

        self.fc1_val = Dense(hidden_1, activation='relu')
        self.fc2_val = Dense(hidden_2, activation='relu')
        self.fc3_val = Dense(1)

        self.fc1_adv = Dense(hidden_1, activation='relu')
        self.fc2_adv = Dense(hidden_2, activation='relu')
        self.fc3_adv = Dense(action_size)

    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs)

        val = self.fc1_val(x)
        val = self.fc2_val(val)
        val = self.fc3_val(val)

        # advantage calculation
        adv = self.fc1_adv(x)
        adv = self.fc2_adv(adv)
        adv = self.fc3_adv(adv)
        return val + adv - k.mean(adv)

    def get_config(self):
        return {
            'name': 'QNetwork',
            'layers': [
                self.fc.get_config(),
                self.fc1_val.get_config(),
                self.fc2_val.get_config(),
                self.fc3_val.get_config(),
                self.fc1_adv.get_config(),
                self.fc2_adv.get_config(),
                self.fc3_adv.get_config()
            ]
        }
