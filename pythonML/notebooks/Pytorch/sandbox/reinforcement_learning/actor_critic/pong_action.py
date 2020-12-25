import gym
import matplotlib.pyplot as plt


def test_image_plot(screen):
    plt.figure()
    plt.imshow(screen,
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


if __name__ == '__main__':
    # 2 - UP
    # 3 DOWN
    #4 up
    #5 down
    # env = gym.make(config.env_name).unwrapped
    env = gym.make("PongNoFrameskip-v4").unwrapped
    print(env.action_space.n)
    print(env.get_action_meanings())
    action = 5  # modify this!
    screen = env.reset()
    test_image_plot(screen)
    for i in range(5):  # repeat one action for five times
        screen = env.step(action)[0]
        test_image_plot(screen)


