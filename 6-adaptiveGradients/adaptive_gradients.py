"""
MNIST dataset trained on a small network, 5 different optimizers
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# set the number of epochs and the batch size
num_epochs = 5

# set the learning rate and batch size#
learning_rate = 0.1
batch_size = 64

# set the choices of optimizers#
optimizer_set = {"SGD": tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False),
                 "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1,
                                                                        epsilon=1e-07),
                 "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0,
                                                                            epsilon=1e-07, centered=False),
                 "Adadelta": tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.95, epsilon=1e-07),
                 "Adam": tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                                  amsgrad=False)}
df = pd.DataFrame(columns=('loss','accuracy','val_loss','val_accuracy'))
# Training via different optimizers and different learning rates and batch sizes#
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

    # train and fit the model, then validate
    print("\n********************* Optimizer=" + str(key) + ", batch_size=" + str(
        batch_size) + ", learning_rate=" + str(learning_rate) + " *********************")
    history = model.fit(x_train,
                        y_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(x_test, y_test))
    df.loc[df.shape[0] + 1] = [history.history['loss'],history.history['accuracy'],history.history['val_loss'],history.history['val_accuracy']]
df.index = ["SGD","Adadelta","RMSprop", "Adagrad", "Adam"]


for i in range(len(optimizer_set)):
    plt.plot(df['loss'][i])
plt.title('Training Loss_lr=' + str(learning_rate) + '_bs=' + str(batch_size))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(df.index, loc='upper right')
plt.savefig('training_loss_lr=' + str(learning_rate) + '_bs=' + str(batch_size) + '.png')
plt.show()