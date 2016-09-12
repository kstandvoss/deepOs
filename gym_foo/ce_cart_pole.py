import gym
import gym_tools as gyt
gyt.init()


env = gym.make('CartPole-v1')
assert len(env.observation_space.shape) == 1

indim  = env.observation_space.shape[0]
outdim = env.action_space.n
policy = gyt.get_linear_policy(indim, outdim)


fitfunc = gyt.get_fitness_function(
    env = env,
    π = policy,
    max_episodes = 1000)


CE = gyt.cross_entropy_method(
    n = 100, # society size
    p = 0.2,
    f = fitfunc,
    iterations = 1000,
    d = (indim+1)*outdim,
    )


for i, (θᵢ, μᵢ, σᵢ) in enumerate(CE):
    
    print('i =',i,'\nθ:',θᵢ,'\nμ:',μᵢ,'\nσ:',σᵢ)
    
    gyt.animate(
        env = env,
        π = policy,
        θ = θᵢ,
        max_episodes=10000)

    if gyt.stop: break

env.close()