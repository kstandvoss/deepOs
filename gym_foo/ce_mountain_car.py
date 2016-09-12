import gym
import gym_tools as gyt
gyt.init()


env = gym.make('MountainCar-v0')
indim  = 2
outdim = env.action_space.n
policy = gyt.get_linear_policy(indim, outdim)


fitfunc = gyt.get_fitness_function(
    env = env,
    π = policy,
    max_episodes = 10000)


CE = gyt.cross_entropy_method(
    n = 100, # society size
    p = 0.2,
    f = fitfunc,
    iterations = 1000,
    d = (indim+1)*outdim,
    )


for i, (θᵢ, μᵢ, σᵢ) in enumerate(CE):
    print('\ni=',i,'\nθ:',θᵢ,'\nμ:',μᵢ,'\nσ:',σᵢ)

    if gyt.stop:
        break
    elif i < 20 or i % 10 != 0:
        continue

    print('\ni=',i,'\nθ:',θᵢ,'\nμ:',μᵢ,'\nσ:',σᵢ)
    
    gyt.animate(
        env = env,
        π = policy,
        θ = θᵢ,
        max_episodes=2000)

    if gyt.stop: break

env.close()