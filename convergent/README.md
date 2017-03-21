# Correlations of Network Layers

Dependencies: Keras + Theano

To create correlations for two non-convolutional networks run *calc_correlations.py*. For convolutional networks pass *cnn* as an argument. Note that reading activations only works with theano backend. You may need to create a *models* and *results* folder. Load a result with the *load_result* function from *experiemt.py*.
