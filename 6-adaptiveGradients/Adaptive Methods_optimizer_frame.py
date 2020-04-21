"""
TensorFlow 2.0
Python 3.7

author: Jiali Zhang
title: MNIST dataset trained on a small network, 5 different optimizers

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# set the number of epochs
num_epochs = 50

# set the learning rate and batch size
learning_rate = 0.01
batch_size = 64

# set the choices of optimizers
optimizer_set = {"SGD": tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False),
                 "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07),
                 "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False),
                 "Adadelta": tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.95, epsilon=1e-07),
                 "Adam": tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)}

# define a data_frame to store the results of the model.
# The column names are defined by the returning of "history" in the model fit
optimizer_frame = pd.DataFrame(columns=('loss','accuracy','val_loss','val_accuracy'))

# Training via different optimizers
for (key,value) in optimizer_set.items():
    # build the neural network model using keras sequencial model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # set the model parameters and optimizer
    model.compile(optimizer=value,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train and fit the model
    print("\n********************* Optimizer=" + str(key) + ", batch_size=" + str(batch_size) + ", learning_rate=" + str(learning_rate) + " *********************")
    """
    when running the SGD optimizer, the number of iterations in each epoch depends on the batch_size.
    for example:
        num_iteration = int(mnist.num_train_data // batch_size)
        937 = 60000 // 64
    """
    MLP =     model.fit(x_train,
                        y_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(x_test, y_test))

    # write the results of each epoch into the frame we defined
    optimizer_frame.loc[optimizer_frame.shape[0] + 1] = [MLP.history['loss'],
                                                         MLP.history['accuracy'],
                                                         MLP.history['val_loss'],
                                                         MLP.history['val_accuracy']]

# modify the row names  of the optimizer_frame via "index"
optimizer_frame.index = ["SGD","Adadelta","RMSprop", "Adagrad", "Adam"]

'''
plt.figure(1)
plt.suptitle('Optimizer_lr=' + str(learning_rate) + '_bs=' + str(batch_size))
plt.subplot(2,2,1)
for i in range(len(optimizer_set)):
    plt.plot(optimizer_frame['loss'][i])
plt.title('Training_Loss')
plt.ylabel('Training_Loss')
plt.xlabel('Epoch')
plt.legend(optimizer_frame.index, loc='upper right')    # row names

plt.subplot(2,2,2)
for i in range(len(optimizer_set)):
    plt.plot(optimizer_frame['accuracy'][i])
plt.title('Training_Accuracy')
plt.ylabel('Training_Accuracy')
plt.xlabel('Epoch')
plt.legend(optimizer_frame.index, loc='upper right')

plt.subplot(2,2,3)
for i in range(len(optimizer_set)):
    plt.plot(optimizer_frame['val_loss'][i])
plt.title('Testing_Loss')
plt.ylabel('Testing_Loss')
plt.xlabel('Epoch')
plt.legend(optimizer_frame.index, loc='upper right')

plt.subplot(2,2,4)
for i in range(len(optimizer_set)):
    plt.plot(optimizer_frame['val_accuracy'][i])
plt.title('Testing_Accuracy')
plt.ylabel('Testing_Accuracy')
plt.xlabel('Epoch')
plt.legend(optimizer_frame.index, loc='upper right')

plt.savefig('Optimizer_' + str(learning_rate) + '_bs=' + str(batch_size) + '.png')
plt.show()
'''

plt.figure(1)
for i in range(len(optimizer_set)):
    plt.plot(optimizer_frame['loss'][i])
plt.title('Training_Loss_lr=' + str(learning_rate) + '_bs=' + str(batch_size))
plt.ylabel('Training_Loss')
plt.xlabel('Epoch')
plt.legend(optimizer_frame.index, loc='upper right')    # row names
plt.savefig('Training_Loss_lr=' + str(learning_rate) + '_bs=' + str(batch_size) + '.png')
plt.show()

plt.figure(2)
for i in range(len(optimizer_set)):
    plt.plot(optimizer_frame['accuracy'][i])
plt.title('Training_Accuracy_lr=' + str(learning_rate) + '_bs=' + str(batch_size))
plt.ylabel('Training_Accuracy')
plt.xlabel('Epoch')
plt.legend(optimizer_frame.index, loc='upper right')
plt.savefig('Training_Accuracy_lr=' + str(learning_rate) + '_bs=' + str(batch_size) + '.png')
plt.show()

plt.figure(3)
for i in range(len(optimizer_set)):
    plt.plot(optimizer_frame['val_loss'][i])
plt.title('Testing_Loss_lr=' + str(learning_rate) + '_bs=' + str(batch_size))
plt.ylabel('Testing_Loss')
plt.xlabel('Epoch')
plt.legend(optimizer_frame.index, loc='upper right')
plt.savefig('Testing_Loss_lr=' + str(learning_rate) + '_bs=' + str(batch_size) + '.png')
plt.show()

plt.figure(4)
for i in range(len(optimizer_set)):
    plt.plot(optimizer_frame['val_accuracy'][i])
plt.title('Testing_Accuracy_lr=' + str(learning_rate) + '_bs=' + str(batch_size))
plt.ylabel('Testing_Accuracy')
plt.xlabel('Epoch')
plt.legend(optimizer_frame.index, loc='upper right')
plt.savefig('Testing_Accuracy_lr=' + str(learning_rate) + '_bs=' + str(batch_size) + '.png')
plt.show()
