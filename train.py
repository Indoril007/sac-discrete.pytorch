import os
import yaml
import argparse
from datetime import datetime
import gym
import numpy as np

from sacd.env import make_pytorch_env, MineRLWrapper
from sacd.agent import SacdAgent, SharedSacdAgent
import minerl


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    # env = make_pytorch_env(args.env_id, clip_rewards=False)
    # test_env = make_pytorch_env(
    #     args.env_id, episode_life=False, clip_rewards=False)

    action_centroids = np.load(os.path.join('./action_centroids.npy'))

    env_name = "MineRLObtainIronPickaxeDenseVectorObf-v0"
    test_env_name = "MineRLObtainIronPickaxeVectorObf-v0"
    env = MineRLWrapper(gym.make(env_name), action_centroids=action_centroids)
    test_env = MineRLWrapper(gym.make(test_env_name), action_centroids=action_centroids)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SacdAgent if not args.shared else SharedSacdAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=int(args.cuda),
        seed=args.seed, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    # parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--cuda', type=str, default="0")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
