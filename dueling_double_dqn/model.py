from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K


class QNetwork(Model):
    def __init__(self, action_size, hidden_1=256, hidden_2=128):
        super(QNetwork, self).__init__()

        self.fc1_val = Dense(hidden_1, activation='relu')
        self.fc2_val = Dense(hidden_2, activation='relu')
        self.fc3_val = Dense(1)

        self.fc1_adv = Dense(hidden_1, activation='relu')
        self.fc2_adv = Dense(hidden_2, activation='relu')
        self.fc3_adv = Dense(action_size)

    def call(self, inputs, training=None, mask=None):
        val = self.fc1_val(inputs)
        val = self.fc2_val(val)
        val = self.fc3_val(val)

        # advantage calculation
        adv = self.fc1_adv(inputs)
        adv = self.fc2_adv(adv)
        adv = self.fc3_adv(adv)
        return val + adv - K.mean(adv)

    def get_config(self):
        return {
            'name': 'QNetwork',
            'layers': [
                self.fc1_val.get_config(),
                self.fc2_val.get_config(),
                self.fc3_val.get_config(),
                self.fc1_adv.get_config(),
                self.fc2_adv.get_config(),
                self.fc3_adv.get_config()
            ]
        }