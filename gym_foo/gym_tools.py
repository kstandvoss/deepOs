
import numpy as np
from numpy.random import multivariate_normal as N
from numpy.random import choice
import gym
import signal


def init():
    np.set_printoptions(precision=3)

    # for gracefull exits
    global stop
    stop = False
    def signal_handler(signal, frame):
        global stop
        stop = True
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to stop!')


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def get_linear_policy(dᵢₙ, dₒᵤₜ):

    def π(s, θ):
        """
        Maps from a state to an action.
        Modeled as a simple perceptron.
        """

        θ = θ.reshape(dₒᵤₜ, dᵢₙ+1)
        s = np.append(1, s)
        #return choice(np.arange(dₒᵤₜ), p=softmax(θ @ s))
        return np.argmax(θ @ s)

    return π


def get_fitness_function(env, π, max_episodes):

    def f(θ):
        R = 0.0 # total reward
        s = env.reset() # initial observation / state

        for i in range(max_episodes):
            a = π(s,θ) 
            s, r, done, info = env.step(a)
            R += r
            if info: print(info)
            if done: break

        return R/np.sqrt(i) # this really depends on the problem

    return f
    
   
def cross_entropy_method(n, f, p, d, iterations):

    # init parameters
    μ = N(np.zeros(d), np.eye(d)*np.ones(d))
    σ = abs(N(np.zeros(d), 10*np.eye(d)*np.ones(d)))

    for i in range(iterations):
        # sample new parameters & evaluate fittness
        θ = N(μ, np.eye(d)*σ, n)
        #import ipdb; ipdb.set_trace()
        
        R = np.array(list(map(f,θ)))
        
        # sort θ by performance and yield best one
        θ = θ[np.argsort(R)]
        yield θ[0], μ, σ

        # fit new parameter distribution best p parameters
        cut_i = int(len(θ)*p)
        elite = θ[cut_i:]
        μ = np.mean(elite, axis=0)
        σ = np.var(elite, axis=0) 


def animate(π, θ, env, max_episodes=1000):
    
    s = env.reset()
    env.render()

    for t in range(max_episodes):
        a = π(s, θ)
        s, r, done, info = env.step(a)
        env.render()
        if info: print(info)
        if done: break

    print("Episode finished after {} timesteps".format(t+1))