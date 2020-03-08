"""
MINIST dataset trained on a small network, different optimizers
"""

#install TensorFlow
import tensorflow as tf
import matplotlib.pyplot as plt

#prepare the MINIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#set the number of epochs and the batch size
num_epochs=100
batch_size=64

#Train via SGD
#build the neural network model using keras sequencial model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
#set the model parameters and optimizer
model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train and fit the model, then validate
print("\n********************* SGD *********************")
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2,validation_data=(x_test, y_test))

"""
dict=['acc', 'loss', 'val_acc', 'val_loss']
['acc'] refers to the accuracy of training set
['val_acc'] refers to the accuracy of validation set
['loss'] refers to the loss of training set
['val_loss'] refers to the loss of validation set
"""
trainacc_SGD=history.history['acc']
trainloss_SGD=history.history['loss']
testacc_SGD=history.history['val_acc']
testloss_SGD=history.history['val_loss']



#Train via Nesterov's momentum
#build the neural network model using keras sequencial model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
#set the model parameters and optimizer
model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=True),
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train and fit the model, then validate
print("\n********************* Nesterov *********************")
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2,validation_data=(x_test, y_test))

trainacc_Nesterov=history.history['acc']
trainloss_Nesterov=history.history['loss']
testacc_Nesterov=history.history['val_acc']
testloss_Nesterov=history.history['val_loss']


#Train via Adagrad
#build the neural network model using keras sequencial model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
#set the model parameters and optimizer
model.compile(optimizer= tf.keras.optimizers.Adagrad(learning_rate=0.01),
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train and fit the model, then validate
print("\n********************* Adagrad *********************")
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2,validation_data=(x_test, y_test))

trainacc_Adagrad=history.history['acc']
trainloss_Adagrad=history.history['loss']
testacc_Adagrad=history.history['val_acc']
testloss_Adagrad=history.history['val_loss']




#plot and compare the accuracies of training set
plt.plot(trainacc_SGD)
plt.plot(trainacc_Nesterov)
plt.plot(trainacc_Adagrad)
plt.title('Training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['SGD', 'Nesterov', 'Adagrad'], loc='lower right')
plt.savefig('test_accuracy.jpg')
plt.show()

#plot and compare the accuracies of validation set
plt.plot(testacc_SGD)
plt.plot(testacc_Nesterov)
plt.plot(testacc_Adagrad)
plt.title('Test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['SGD', 'Nesterov', 'Adagrad'], loc='lower right')
plt.savefig('test_accuracy.jpg')
plt.show()

