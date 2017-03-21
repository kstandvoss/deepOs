import os

from pylab import *
import pickle

import experiment

result = experiment.load_result()

print('plot %s layers' % len(result['between_net_corr']))

def enumzip(*args, start=0):
    yield from enumerate(zip(*args), start=0)

layers = result['res1']['layers']
between_corr = result['between_net_corr']

figure('between_net_corr')
for i, (lay, bnc) in enumzip(layers, ):
    subplot(3,3,i+1)
    imshow(bnc, cmap='magma')
    #hist(bnc)
    xticks([])
    yticks([])
    title('bnc - %s - %s' % (i, lay))
colorbar()


figure('within_net_corr_1')
within_corr_1 = result['res1']['stats']['within_corr']
for i, (lay, bnc) in enumzip(layers, within_corr_1):
    subplot(3,3,i+1)
    imshow(bnc)
    #hist(bnc)
    xticks([])
    yticks([])
    title('bnc - %s - %s' % (i, lay))
colorbar()

figure('within_net_corr_2')
within_corr_2 = result['res1']['stats']['within_corr']
for i, (lay, bnc) in enumzip(layers, within_corr_2):
    subplot(3,3,i+1)
    imshow(bnc, cmap='RdBu')
    #hist(bnc)
    xticks([])
    yticks([])
    title('bnc - %s - %s' % (i, lay))
colorbar()

# take random thin



squeeze = lambda L : [mean(x) for x in L]

mean1 = squeeze(result['res1']['stats']['mean'])
std1 = squeeze(result['res1']['stats']['std'])
mean2 = squeeze(result['res2']['stats']['mean'])
std2 = squeeze(result['res2']['stats']['std'])

"""layers = results['model'].layers
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
"""
show()
