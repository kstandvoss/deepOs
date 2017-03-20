from time import time

import numpy as np

from keras_mnist import load_mnist, build_model
from keras_mnist_cnn import load_mnist_cnn, build_model_cnn
import experiment


def calculate_two_models(cnn):
    res1 = calculate_collect_activations('model1', cnn)
    res2 = calculate_collect_activations('model2', cnn)
    between_net_corr = experiment.calc_between_net_corr(
        acts1 = res1['activations'],
        acts2 = res2['activations'],
    )
    return locals()


def calculate_collect_activations(name, cnn):
    if cnn:
        (X_train, y_train), (X_test, y_test) = load_mnist_cnn()
    else:
    (X_train, y_train), (X_test, y_test) = load_mnist()

    if cnn:
        name += '_cnn'
    name += '.h5'
    tic = time()
    try:
        model = keras.models.load_model(name)
        print('load model', end='\t', flush=True)
    except:
        print('create model')
        if cnn:
            model = build_cnn_model()
        else:
            model = build_model()
        model.fit(X_train, y_train,
            batch_size=128, nb_epoch=4, verbose=1,
            validation_data=(X_test, y_test))
        model.save(name)
    print(round(time() - tic), 'secs')

    print('calc score', end='\t', flush=True)
    tic = time()
    score = model.evaluate(X_test, y_test, verbose=0)
    print(round(time() - tic), 'secs')

    print('calc activation', end='\t', flush=True)
    tic = time()
    activations = experiment.get_all_outputs(model, X_test)
    print(round(time() - tic), 'secs')

    print('calc stats', end='\t', flush=True)
    tic = time()
    stats = experiment.calc_single_act_stats(activations)
    print(round(time() - tic))

    del tic, model
    return locals()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cnn':
            result = calculate_two_models(cnn=True)
        else:
            raise Exception('invalid argument' + sys.argv[1])
        filename = 'result_' + sys.argv[0] + '_'
    else:
        result = calculate_two_models(cnn=True)
        filename = 'result_'

    experiment.save_result(result, filename + str(round(time())))
