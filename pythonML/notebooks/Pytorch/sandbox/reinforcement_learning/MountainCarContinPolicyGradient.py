# https://github.com/openai/gym/wiki/MountainCar-v0
import gym

env = gym.make('MountainCarContinuous-v0')
state1 = env.reset()
action = env.action_space.sample()
state, reward, done, info = env.step(action)
a = 0
for _ in range(2000):
    env.render()
    act = env.action_space.sample()
    print(act)
    state, reward, done, info = env.step(act)
    print("state = {0} reward = {1} done = {2} info = {3}".format(state, reward, done, info))

env.close()