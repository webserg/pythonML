from A2CAgent import A2CAgent

if __name__ == '__main__':
    training = True

    env_id = 'CartPole-v1'
    value_learning_rate = 0.001
    actor_learning_rate = 0.001
    gamma = 0.99

    config_a2c = {
        'env_id': env_id,
        'gamma': gamma,
        'value_network': {'learning_rate': value_learning_rate},
        'actor_network': {'learning_rate': actor_learning_rate},
        'file_path_actor': '../../models/a2cActorCartPole.pt',
        'file_path_critic': '../../models/a2cCriticCartPole.pt'
    }

    print("Current config_a2c is:")
    print(config_a2c)

    agent = A2CAgent(config_a2c)
    if training:
        rewards = agent.training_batch(1000, 256)
    agent.evaluate(True)