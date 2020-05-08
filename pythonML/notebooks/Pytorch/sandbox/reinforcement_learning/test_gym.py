#https://github.com/openai/gym/wiki/CartPole-v0
import gym

env = gym.make('CartPole-v0')
state1 = env.reset()
action = env.action_space.sample()
state, reward, done, info = env.step(action)

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()