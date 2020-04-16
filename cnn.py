import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.api._v2 import keras
from tensorflow_core.python.keras.api._v2.keras.datasets import mnist 
from tensorflow_core.python.keras.api._v2.keras import layers

keras.backend.clear_session()


class SoftmaxLayer(layers.Layer):
    def __init__(self, input_size, output_size, initializer):
        super(SoftmaxLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_size, output_size),
            initializer=initializer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(output_size,),
            initializer=keras.initializers.constant(0.1),
            trainable=True
        )

    def __call__(self, inputs):
        return tf.nn.softmax(tf.linalg.matmul(inputs, self.w) + self.b)


class ReluLayer(layers.Layer):
    def __init__(self, input_size, output_size, initializer):
        super(ReluLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_size, output_size),
            initializer=initializer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(output_size,),
            initializer=keras.initializers.constant(0.1),
            trainable=True
        )
    
    def __call__(self, inputs):
        return tf.nn.relu(tf.linalg.matmul(inputs, self.w) + self.b)


class ConvolutionLayer(layers.Layer):
    def __init__(self, height, width, input_dim, output_dim, initializer):
        super(ConvolutionLayer, self).__init__()
        self.w = self.add_weight(
            shape=(height, width, input_dim, output_dim),
            initializer=initializer,
            trainable=True
        )
        self.b = self.add_weight(
            shape=(output_dim,),
            initializer=keras.initializers.constant(0.1),
            trainable=True
        )

    def __call__(self, inputs, strides, padding):
        temp = tf.nn.conv2d(inputs, self.w, strides=strides, padding=padding) + self.b
        return tf.nn.relu(temp)


class MaxPoolLayer(layers.Layer):
    def __init__(self):
        super(MaxPoolLayer, self).__init__()

    def __call__(self, inputs, ksize, strides, padding):
        return tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding=padding)


class FlattenLayer(layers.Layer):
    def __init__(self, size, channels_dim):
        super(FlattenLayer, self).__init__()
        self.size = size
        self.channels_dim = channels_dim

    def __call__(self, inputs):
        return tf.reshape(inputs, [-1, self.size**2 * self.channels_dim])


class DropoutLayer(layers.Layer):
    def __init__(self, probability):
        super(DropoutLayer, self).__init__()
        self.probability = probability

    def __call__(self, inputs):
        return tf.nn.dropout(inputs, rate=self.probability)


class ConvolutionModel(keras.Model):
    def __init__(self):
        super(ConvolutionModel, self).__init__()
        self.conv_layer1 = ConvolutionLayer(5, 5, 1, 32, keras.initializers.TruncatedNormal(stddev=0.1))
        self.maxpool_layer = MaxPoolLayer()
        self.conv_layer2 = ConvolutionLayer(5, 5, 32, 64, keras.initializers.TruncatedNormal(stddev=0.1))
        self.flatten_layer = FlattenLayer(7, 64)
        self.relu_layer = ReluLayer(7*7*64, 1024, keras.initializers.TruncatedNormal(stddev=0.1))
        self.dropout_layer = DropoutLayer(0.5)
        self.softmax_layer = SoftmaxLayer(1024, 10, keras.initializers.TruncatedNormal(stddev=0.1))
    
    def __call__(self, inputs):
        inputs = self.conv_layer1(inputs, [1, 1, 1, 1], "SAME")
        inputs = self.maxpool_layer(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        inputs = self.conv_layer2(inputs, [1, 1, 1, 1], "SAME")
        inputs = self.maxpool_layer(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        inputs = self.flatten_layer(inputs)
        inputs = self.relu_layer(inputs)
        inputs = self.dropout_layer(inputs)
        return self.softmax_layer(inputs)


batch_size = 64
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = ConvolutionModel()
loss_func = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)
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
