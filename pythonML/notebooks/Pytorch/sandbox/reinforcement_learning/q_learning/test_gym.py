# https://github.com/openai/gym/wiki/CartPole-v0
import gym
from gym import envs

# env = gym.make('CartPole-v0')
# env = gym.make('Pong-v0')
env = gym.make('MsPacman-v0')
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('Pendulum-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('CarRacing-v0')
# env = gym.make('Alien-v0')
# env = gym.make('Kangaroo-v0')
# env = gym.make('Riverraid-v0')
# env = gym.make('FrozenLake-v0')
# env = gym.make('Breakout-v0')
# env = gym.make('Ant-v2')
# env = gym.make('Snake-v0')
# env = gym.make('Taxi-v3')
# env = gym.make('LunarLander-v2')
# env = gym.make('Robotank-v0')
state1 = env.reset()
action = env.action_space.sample()
print(env.action_space)
print(env.observation_space)
print(env.reward_range)
state, reward, done, info = env.step(action)
print("state = {0} reward = {1} done = {2} info = {3}".format(state, reward, done, info))
a = 0
for _ in range(200):
    env.render()
    act = env.action_space.sample()
    print(act)
    state, reward, done, info = env.step(act)
    if done:
        env.reset()
    print("state = {0} reward = {1} done = {2} info = {3}".format(state.shape, reward, done, info))

env.close()


envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)
