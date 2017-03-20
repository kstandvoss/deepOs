from pylab import *
import pickle

results = pickle.load(open('result_1489764752', 'rb'))
layers = results['model'].layers
means = results['stats']['mean']
stds = results['stats']['std']

figure('mean per layer')
for i,(lay, m) in enumerate(zip(layers, means)):
    subplot(3,2,i+1)
    plot(m)
    title(lay.name)

figure('std per layer')
for i,(lay, s) in enumerate(zip(layers, stds)):
    subplot(3,2,i+1)
    plot(s)
    title(lay.name)

show()
