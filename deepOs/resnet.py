import os

import keras
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot
import datetime
from deepOs.statefarm import Statefarm

# see http://arxiv.org/pdf/1512.03385v1.pdf
class Resnet:

    def __init__(self, 
                 statefarm,
                 params=None):
        self.statefarm = statefarm
        self.params = Resnet.parameters()
        if params is not None:
            self.params.update(params)
        self.model = self.init_model()
        

    # Helper to build a conv -> BN -> relu block
    def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
        def f(input):
            conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                                 init="he_normal", border_mode="same")(input)
            norm = BatchNormalization(mode=0, axis=1)(conv)
            return Activation("relu")(norm)

        return f


    # Helper to build a BN -> relu -> conv block
    # This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
        def f(input):
            norm = BatchNormalization(mode=0, axis=1)(input)
            activation = Activation("relu")(norm)
            conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                                 init="he_normal", border_mode="same")(activation)
            return conv

        return f


    # Bottleneck architecture for > 34 layer resnet.
    # Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    # Returns a final conv layer of nb_filter * 4
    def _bottleneck(nb_filter, init_subsample=(1, 1)):
        def f(input):
            conv_1_1 = Resnet._bn_relu_conv(nb_filter, 1, 1, subsample=init_subsample)(input)
            conv_3_3 = Resnet._bn_relu_conv(nb_filter, 3, 3)(conv_1_1)
            residual = Resnet._bn_relu_conv(nb_filter * 4, 1, 1)(conv_3_3)
            shortcut = Resnet._shortcut(input, residual)
            return shortcut

        return f


    # Basic 3 X 3 convolution blocks.
    # Use for resnet with layers <= 34
    # Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    def _basic_block(nb_filter, init_subsample=(1, 1)):
        def f(input):
            conv1 = Resnet._bn_relu_conv(nb_filter, 3, 3, subsample=init_subsample)(input)
            residual = Resnet._bn_relu_conv(nb_filter, 3, 3)(conv1)
            return Resnet._shortcut(input, residual)

        return f


    # Adds a shortcut between input and residual block and merges them with "sum"
    def _shortcut(input, residual):
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        stride_width = int(input._keras_shape[2] / residual._keras_shape[2])
        stride_height = int(input._keras_shape[3] / residual._keras_shape[3])
        equal_channels = residual._keras_shape[1] == input._keras_shape[1]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                     subsample=(stride_width, stride_height),
                                     init="he_normal", border_mode="valid")(input)

        return merge([shortcut, residual], mode="sum")


    # Builds a residual block with repeating bottleneck blocks.
    def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
        def f(input):
            for i in range(repetitions):
                init_subsample = (1, 1)
                if i == 0 and not is_first_layer:
                    init_subsample = (2, 2)
                input = block_function(nb_filter=nb_filter, init_subsample=init_subsample)(input)
            return input

        return f

    def init_model(self):
        input = Input(shape=(self.params['input_colors'], self.params['input_dim0'], self.params['input_dim1']))

        conv0 = Resnet._conv_bn_relu(nb_filter=self.params['nb_filter0'], 
                              nb_row=self.params['nb_row0'], 
                              nb_col=self.params['nb_col0'], 
                              subsample=(2, 2))(input)
        pool0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv0)

        # Build residual blocks..
        block_fn = Resnet._bottleneck
        block1 = Resnet._residual_block(block_fn, nb_filter=self.params['nb_filter1'], repetitions=self.params['nb_repetitions1'], is_first_layer=True)(pool0)
        block2 = Resnet._residual_block(block_fn, nb_filter=self.params['nb_filter2'], repetitions=self.params['nb_repetitions2'])(block1)
        block3 = Resnet._residual_block(block_fn, nb_filter=self.params['nb_filter3'], repetitions=self.params['nb_repetitions3'])(block2)
        block4 = Resnet._residual_block(block_fn, nb_filter=self.params['nb_filter4'], repetitions=self.params['nb_repetitions4'])(block3)

        # Classifier block
        pool5 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4)
        flatten5 = Flatten()(pool5)
        
        dense = Dense(output_dim=10, init="he_normal", activation="softmax")(flatten5)

        
        self.model = Model(input=input, output=dense)
        
        print('Compile model')
        optim = keras.optimizers.RMSprop(lr=self.params['lr'], rho=0.9, epsilon=1e-06)
        
        self.model.compile(loss=Statefarm.log_loss_objective, optimizer=optim, metrics=["accuracy"])
    
        return self.model
    
    def parameters():
        return {
            'batch_size': 64,
            'nb_epoch': 100,
            'lr': 0.01,
            'input_colors': 1,
            'input_dim0': 224,
            'input_dim1': 224,
            'nb_filter0': 64,
            'nb_row0': 7,
            'nb_col0': 7,
            'nb_filter1': 64,
            'nb_filter2': 128,
            'nb_filter3': 256,
            'nb_filter4': 512,
            'nb_repetitions1': 3,
            'nb_repetitions2': 4,
            'nb_repetitions3': 6,
            'nb_repetitions4': 3,
        }
    
    def train(self, train_driver_ids, valid_driver_ids):
        
        train_data, train_target, driver_id, unique_drivers = self.statefarm.read_and_normalize_train_data(
            self.params['input_dim0'], 
            self.params['input_dim1'], 
            self.params['input_colors'])
        
        drivers = ['p002', 'p012', 'p014', 'p015', 'p021', 'p022', 'p024',
                 'p026', 'p035', 'p039', 'p041', 'p045', 'p047', 'p049',
                 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072', 'p081', 'p042', 'p075', 'p016']
        
        drivers_train = [drivers[i] for i in train_driver_ids]
        drivers_valid = [drivers[i] for i in valid_driver_ids]
        
        print('copy train data')
        X_train, Y_train, train_index = self.statefarm.copy_selected_drivers(train_data, train_target, driver_id, drivers_train)
        print('copy valid data')
        X_valid, Y_valid, valid_index = self.statefarm.copy_selected_drivers(train_data, train_target, driver_id, drivers_valid)    
        del train_data
        
        #init_log_loss_valid = self.model.evaluate(X_valid, Y_valid, verbose=0)
        #print('Initial: log_loss on validation set: ' + str(init_log_loss_valid[0]))

        print('Define training callbacks to save checkpoints')
        now = datetime.datetime.now()
        checkpoint_filename = 'checkpoint_' + str(now.strftime("%Y-%m-%d-%H-%M-%S")) + '.hdf5'
        checkpoint_path = os.path.join('checkpoints', checkpoint_filename)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

        print('Start training')
        history = self.model.fit(X_train, Y_train, 
                            batch_size=self.params['batch_size'], 
                            nb_epoch=self.params['nb_epoch'], 
                            verbose=1, 
                            validation_data=(X_valid, Y_valid), 
                            callbacks=[checkpoint_callback, early_stopping_cb])

        print('early stopping triggered after iteration: ' + str(len(history.history['loss'])))

        epochId_best_validation = np.argmin(history.history["val_loss"])
        print('epoch with best performance on validation set: ' + str(epochId_best_validation))

        print('Reload best validation weights')
        self.model.load_weights(checkpoint_path)

        predictions_valid = model.predict(X_valid, batch_size=self.params['batch_size'], verbose=0)
        log_loss_valid_per_sample = log_loss_objective_compiled(Y_valid, predictions_valid.astype('float32'))
        
        return history, predictions_valid, log_loss_valid_per_sample