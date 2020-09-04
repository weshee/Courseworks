# Course 1: Introduction to TensorFlow for Artificial Intelligence, Machine Learning and Deep Learning
import tensorflow as tf
import numpy as np
from tensorflow import keras

# General workflow
def simple_tf():
    # 1. Define and compile a simple model
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    mode.compile(optimizer='sgd', loss='mean_squared_error')
    # 2. Providing the data
    xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    # 3. Training the nn
    model.fit(xs, ys, epochs=500)
    # 4. Prediction
    print(model.predict([10.0]))

# Simple neural network
def simple_nn():
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
          print("\nReached 60% accuracy so cancelling training!")
          self.model.stop_training = True
    callbacks = myCallback()
    # Stop the training when a desired value is reached
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels),  (test_images, test_labels) = mnist.load_data()
    training_images = training_images/255.0
    test_images = test_images/255.0
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    # the first layer in your network should be the same shape as your data
    # the number of neurons in the last layer should match the number of classes you are classifying for
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])

# Simple CNN
def simple_cnn():
    import tensorflow as tf
    print(tf.__version__)
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(training_images, training_labels, epochs=5)
    test_loss = model.evaluate(test_images, test_labels)

    def visualizing_conv_pooling():
        import matplotlib.pyplot as plt
        f, axarr = plt.subplots(3, 4)
        FIRST_IMAGE = 0
        SECOND_IMAGE = 23
        THIRD_IMAGE = 28
        CONVOLUTION_NUMBER = 25
        from tensorflow.keras import models
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        for x in range(0, 4):
            f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
            axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
            axarr[0, x].grid(False)
            f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
            axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
            axarr[1, x].grid(False)
            f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
            axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
            axarr[2, x].grid(False)

# For image processing
def cnn_image():
    # Data extracted from horse-or-human.zip are not shown here
    # Sequential modelling not shown here
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1 / 255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        class_mode='binary') # Since we use binary_crossentropy loss, we need binary labels
    history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1,
                        validation_data=validation_generator, validation_steps=8)
    # Visualizing intermediate representation is available in the notebook, not shown here

# Evaluating accuracy and loss for the model
def eval():
    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))  # Get number of epochs
    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.figure()
    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

# Course 2: Convolutional Neural Networks in TensorFlow
# Data augmentation
def data_aug():
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

# Transfer learning
def transfer_learning():
    import os
    from tensorflow.keras import layers
    from tensorflow.keras import Model
    from tensorflow.keras.optimizers import RMSprop
    #!wget - -no - check - certificate \
    #        https: // storage.googleapis.com / mledu - datasets / inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    #                  - O / tmp / inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
    pre_trained_model.load_weights(local_weights_file)
    for layer in pre_trained_model.layers:
        layer.trainable = False

    pre_trained_model.summary()
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)
    model = Model(pre_trained_model.input, x)
    model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy',
                  metrics=['accuracy'])
    # history = model.fit(...)

# Running training samples in batches
def get_data(filename):
    with open(filename) as training_file:
        f = training_file.readlines()
        columns = np.array(f[0].strip('\n').split(','))  # not using
        labels = []
        images = []
        idx = 1
        for idx in range(len(f)-1):
            row_list = f[idx + 1].strip('\n').split(',')
            row = np.array([int(i) for i in row_list])
            row_label = row[0]
            row_image = row[1:].reshape(28,-1)
            labels.append(row_label)
            images.append(row_image)
        labels = np.array(labels).astype(int)
        images = np.array(images).astype(float)
    return images, labels

def image_generator():
    path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
    path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
    training_images, training_labels = get_data(path_sign_mnist_train)
    testing_images, testing_labels = get_data(path_sign_mnist_test)
    training_images = training_images[:,:,:,np.newaxis]
    testing_images = testing_images[:,:,:,np.newaxis]
    train_datagen = ImageDataGenerator(
        rescale = 1./255, rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1/255)

def model_batches():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    # Compile Model.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the Model
    history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=32), epochs=3,
                                  steps_per_epoch=len(training_images) / 32,
                                  validation_data=validation_datagen.flow(testing_images, testing_labels,
                                                                          batch_size=32),
                                  verbose=1, validation_steps=len(testing_images) / 32)
    # Your Code Here (set 'epochs' = 2))

    model.evaluate(testing_images, testing_labels, verbose=0)
    def plot_accuracy_chart():
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_accuracy']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

# Course 3: NLP in TensorFlow
def tokenizer():
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sentences = [...]  # 'I love my dog', 'I love my cat', 'You love my dog!'
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=5, padding='post')
    # output:
    # Padded Sequences:
    # [[ 0  5  3  2  4]
    #  [ 0  5  3  2  7]
    #  [ 0  6  3  2  4]
    #  [ 9  2  4 10 11]]
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_word_index = label_tokenizer.word_index
    label_seq = label_tokenizer.texts_to_sequences(labels)

def NLP_simple_model():
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)  # shape: (vocab_size, embedding_dim)

# Single layer LSTM
def single_LSTM():
    from __future__ import absolute_import, division, print_function, unicode_literals
    import tensorflow_datasets as tfds
    import tensorflow as tf
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    tokenizer = info.features['text'].encoder
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        # For multilayer LSTM
        #   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        #   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # For Conv1D
        #   tf.keras.layers.Conv1D(128, 5, activation='relu'),
        #   tf.keras.layers.GlobalAveragePooling1D(),
        # For GRU
        # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    NUM_EPOCHS = 10
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

    import matplotlib.pyplot as plt
    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')

# Module 4: Sequences, Time Series and Prediction
def dataset():
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.window(5, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(5))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.shuffle(buffer_size=10)
    # [4 5 6 7] [8]
    # [0 1 2 3] [4]
    # [1 2 3 4] [5]
    # [3 4 5 6] [7]
    # [5 6 7 8] [9]
    # [2 3 4 5] [6]
    dataset = dataset.batch(2).prefetch(1)
    for x, y in dataset:
        print("x = ", x.numpy())
        print("y = ", y.numpy())
    # x = [[4 5 6 7]
    #      [5 6 7 8]]
    # y = [[8]
    #      [9]]
    # x = [[3 4 5 6]
    #      [2 3 4 5]]
    # y = [[7]
    #      [6]]
    # x = [[1 2 3 4]
    #      [0 1 2 3]]
    # y = [[5]
    #      [4]]