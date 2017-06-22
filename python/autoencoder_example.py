# started from here:
# https://stackoverflow.com/questions/40554887/stacked-autoencoder-for-classification-using-mnist

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np

nb_classes = 10
nb_epoch = 5
batch_size = 256
hidden_layer1 = 128
hidden_layer2 = 64
hidden_layer3 = 10 # because you have 10 categories
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print('Train samples: {}'.format(x_train.shape[0]))
print('Test samples: {}'.format(x_test.shape[0]))

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

input_img = Input(shape=(784,))
encoded = Dense(hidden_layer1, activation='relu')(input_img)
encoded = Dense(hidden_layer2, activation='relu')(encoded)
encoded = Dense(hidden_layer3, activation='softmax')(encoded)
decoded = Dense(hidden_layer2, activation='relu')(encoded)
decoded = Dense(hidden_layer1, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# first let's set up the autoencoder to see how well it encodes itself
print('Training the autoencoder...')
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(x_train, x_train,
                epochs=5, # 50
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
# score x_test vs itself
auto_score = autoencoder.evaluate(x_test, x_test, verbose=1)
print('\nTest autoencoder loss:', auto_score[0])
print('\nTest autoencoder accuracy:', auto_score[1])
print('\nAutoencoder metrics : ' + str(autoencoder.metrics_names))
				
# and then we'll see how well we can do on the classification task
print('Now training the classification task...')
model = Model(inputs=[input_img], outputs=[encoded])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=nb_epoch,
          batch_size=batch_size,
          shuffle=True,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)

print('/n')
print('Classification Test loss:', score[0])
print('Classification Test accuracy after fine turning:', score[1])

print('Done with script')