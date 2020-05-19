# https://github.com/openai/gym/wiki/CartPole-v0
import gym

# env = gym.make('Pong-v0')
# env = gym.make('MsPacman-v0')
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('Pendulum-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('Alien-v0')
env = gym.make('Kangaroo-v0')
state1 = env.reset()
action = env.action_space.sample()
state, reward, done, info = env.step(action)
a = 0
for _ in range(2000):
    env.render()
    act = env.action_space.sample()
    print(act)

    env.step(act)

env.close()
