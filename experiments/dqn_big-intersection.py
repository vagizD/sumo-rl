import os
import sys

import gymnasium as gym

os.environ["SUMO_HOME"] = "/usr/share/sumo"
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from stable_baselines3.dqn.dqn import DQN
from sumo_rl.models import MyDQN

from sumo_rl import SumoEnvironment


env = SumoEnvironment(
    net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
    single_agent=True,
    route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
    # out_csv_name="outputs/big-intersection/dqn",
    out_csv_name="outputs/big-intersection/my-dqn",
    use_gui=False,
    num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

model = MyDQN(
    env=env,
    # policy="MlpPolicy",
    learning_rate=1e-3,
    buffer_size=50000,
    train_freq=1,
    verbose=1,
    hid_size=256
)
model.learn(total_timesteps=20_000)
