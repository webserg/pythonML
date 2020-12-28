from A2CAgent import A2CAgent

if __name__ == '__main__':

    training = True

    env_id = 'Acrobot-v1'  # @param ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    value_learning_rate = 0.001
    actor_learning_rate = 0.001
    gamma = 0.99

    config_a2c = {
        'env_id': env_id,
        'gamma': gamma,
        'value_network': {'learning_rate': value_learning_rate},
        'actor_network': {'learning_rate': actor_learning_rate},
        'file_path_actor': '../../models/a2cActorAcroBot.pt',
        'file_path_critic': '../../models/a2cCriticAcroBot.pt'
    }

    print("Current config_a2c is:")
    print(config_a2c)

    agent = A2CAgent(config_a2c)
    if training:
        rewards = agent.training_batch(1000, 256)

    for _ in range(10):
        agent.evaluate(True, not training)

    agent.close_env()
