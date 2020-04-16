import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.api._v2 import keras
from tensorflow_core.python.keras.api._v2.keras.datasets import mnist 
from tensorflow_core.python.keras.api._v2.keras import layers

keras.backend.clear_session()


class SoftmaxLayer(layers.Layer):
    def __init__(self, input_size, output_size):
        super(SoftmaxLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_size, output_size),
            initializer="zeros",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(output_size,),
            initializer="zeros",
            trainable=True
        )

    def __call__(self, inputs):
        return tf.nn.softmax(tf.linalg.matmul(inputs, self.w) + self.b)


class TanhActivateLayer(layers.Layer):
    def __init__(self, input_size, output_size, initializer):
        super(TanhActivateLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_size, output_size),
            initializer=initializer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(output_size,),
            initializer=initializer,
            trainable=True
        )
    
    def __call__(self, inputs):
        return tf.nn.tanh(tf.linalg.matmul(inputs, self.w) + self.b)


class DropoutLayer(layers.Layer):
    def __init__(self, probability):
        super(DropoutLayer, self).__init__()
        self.probability = probability

    def __call__(self, inputs):
        return tf.nn.dropout(inputs, rate=self.probability)


class BatchNormLayer(layers.Layer):
    def __init__(self, size):
        super(BatchNormLayer, self).__init__()
        self.scale = self.add_weight(
            shape=(size,),
            initializer="ones",
            trainable=True
        )
        self.beta = self.add_weight(
            shape=(size,),
            initializer="zeros",
            trainable=True
        )

    def __call__(self, inputs):
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, self.beta, self.scale, 0.001)


class FullyconnectedModel(keras.Model):
    def __init__(self, input_size, latent_size, output_size):
        super(FullyconnectedModel, self).__init__()
        self.tanh_layer1 = TanhActivateLayer(input_size, latent_size, "glorot_normal")
        self.batchnorm_layer = BatchNormLayer(latent_size)
        self.tanh_layer2 = TanhActivateLayer(latent_size, latent_size, "glorot_normal")
        self.tanh_layer3 = TanhActivateLayer(latent_size, latent_size, "glorot_normal")
        self.dropout_layer = DropoutLayer(0.1)
        self.output_layer = SoftmaxLayer(latent_size, output_size)
    
    def __call__(self, inputs):
        inputs = self.tanh_layer1(inputs)
        inputs = self.batchnorm_layer(inputs)
        inputs = self.tanh_layer2(inputs)
        inputs = self.tanh_layer3(inputs)
        inputs = self.dropout_layer(inputs)
        return self.output_layer(inputs)


batch_size = 64
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

model = FullyconnectedModel(28*28, 100, 10)
loss_func = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_func(labels, predictions)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_loss(loss)
    train_accuracy(labels, predictions)
    
    
@tf.function
def test_step(inputs, labels):
    predictions = model(inputs)
    loss = loss_func(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


test_acc_plots = []
test_loss_plots = []
epochs = range(10)
for epoch in epochs:
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for inputs, label in train_dataset:
        train_step(inputs, label)
    
    for inputs, label in test_dataset:
        test_step(inputs, label)

    test_acc_plots.append(test_accuracy.result())
    test_loss_plots.append(test_loss.result())
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            test_loss.result(),
                            test_accuracy.result() * 100))


fig1 = plt.figure(1)
plt.plot(epochs, test_acc_plots, "b")
plt.ylabel("Test Accuracy")
plt.xlabel("Epochs")
plt.grid(True)
fig1.suptitle("Accuracy")

fig2 = plt.figure(2)
plt.plot(epochs, test_loss_plots, "r")
plt.ylabel("Test Loss")
plt.xlabel("Epochs")
plt.grid(True)
fig2.suptitle("Loss")

plt.show()
