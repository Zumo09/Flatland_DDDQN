from collections import defaultdict
from tensorboard_logger import configure, log_value
import numpy as np

# TODO Copiato e incollato, non so come funziona

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger
        self.tb_logger = None

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        configure(directory_name)
        self.tb_logger = log_value

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))
        self.tb_logger(key, value, t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)
