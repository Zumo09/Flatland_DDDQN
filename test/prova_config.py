import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense

# in_ = Input(12)
# x = Dense(53, activation='relu')(in_)
# out = Dense(10, activation='relu')(x)
#
# model = Model(inputs=in_, outputs=out)

model = Sequential([
    Dense(53, activation='relu', input_shape=(12,)),
    Dense(10, activation='softmax')
])

model.summary()

cfg = model.get_config()

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(cfg)