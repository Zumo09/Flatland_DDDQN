from argparse import Namespace
import numpy as np
from components.bdddqn_policy import BDDDQNPolicy

policy = BDDDQNPolicy(None, Namespace(num_heads=4), evaluation_mode=True)
policy.load('./checkpoints/210923213346/1500')

head_0 = policy.local_networks[0]

layer = head_0.get_layers()[1]
print(layer)
