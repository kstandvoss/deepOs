import json
import os
import theano
import numpy as np
import keras

"""def save_result(res, name='results'):
    filename = name + '.json'
    if os.path.exists(filename):
        with open(filename,'r') as f:
            list_ = json.load(f)
    else:
        list_ = []
    list_.append(res)
    with open('results.json','w') as f:
        list_ = json.dump(res, f)"""

def save_result(res, name='results'):
    import pickle
    with open(name, 'wb') as f:
        pickle.dump(res, f)

def calc_act_stats(acts1, acts2):
    stats1 = calc_single_act_stats(acts1)
    stats2 = calc_single_act_stats(acts2)

    between_corr = calc_between_net_corr(acts1, acts2)

    del acts1, acts2
    return locals()


def calc_single_act_stats(acts):
    # layer x unit x batch => mean: layer x unit
    # list x array x array => list x array
    mean = [np.mean(la, axis=1) for la in acts]
    std = [np.std(la, axis=1) for la in acts]

    within_corr = calc_within_net_corr(acts)

    del acts
    return locals()


def calc_within_net_corr(acts):
    return calc_between_net_corr(acts, acts)


def calc_between_net_corr(acts1, acts2):
    """
    Calculate the correlation matrix for each layer between two neural networks.

    Li, Yixuan, et al. "Convergent Learning: Do different neural networks learn
    the same representations?." Proceedings of International Conference on
    Learning Representation (ICLR). 2016.
    """

    assert len(acts1) == len(acts2)

    mean1 = [np.mean(la, axis=1) for la in acts1]
    std1 = [np.std(la, axis=1) for la in acts1]
    mean2 = [np.mean(la, axis=1) for la in acts2]
    std2 = [np.std(la, axis=1) for la in acts2]

    corr_per_layer = [] # l x i x j

    variables = zip(acts1, acts2, mean1, mean2, std1, std2)
    for layer_act1, layer_act2, m1, m2, s1, s2 in variables:
        corr_array = np.empty((len(layer_act1), len(layer_act2)))
        for i,xi in enumerate(layer_act1):
            for j,xj in enumerate(layer_act2):
                tmp = (xi - m1[i])*(xj - m2[j])
                corr = np.mean(tmp) / (s1[i]*s2[j])
                corr_array[i,j] = corr
        corr_per_layer.append(corr_array)

    return corr_per_layer


def get_output(model, layer, X_batch):
    """
    Returns the output of a layer from a model given a list of input vectors.

    Parameters
    ----------
    model :
        Sequential keras model
    layer : int
        index of layer
    X_batch : ndarray
        list of input samples

    Returns
    -------
    output :
        output(activation) of each neuron of the layer
    """

    get_output = theano.function(
        [model.layers[0].input, keras.backend.learning_phase()],
        model.layers[layer].output,
        allow_input_downcast=True,
        on_unused_input='ignore',
    )
    return get_output(X_batch, 0) # same result as above


def get_all_outputs(model, X_batch):
    """
    Calls get_output for each layer of a sequential model.
    """

    outputs = []
    for i in range(len(model.layers)):
        act = get_output(model, i, X_batch)
        outputs.append(act)

    # layer x batch x unit => layer x unit x batch
    return [a.T for a in outputs]
