"""
Keras implementation of Auxiliary Classifier GAN
"""

from keras.datasets import cifar10, mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
from data_loader import UTKFace_data
import numpy as np


class ACGAN():
    """Implementation of Deep convolutional GAN"""

    def __init__(self, rows, cols, channels, num_classes, dataset):
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.num_classes = num_classes
        self.dataset = dataset

        optimizer = Adam(0.0002, 0.5)
        losses = [keras.losses.binary_crossentropy, keras.losses.sparse_categorical_crossentropy]

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(100,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(384 * 2 * 2, activation="relu", input_dim=100))
        model.add(Reshape((2, 2, 384)))

        model.add(Conv2DTranspose(192, kernel_size=5, strides=2))
        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(96, kernel_size=3, strides=2))
        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2DTranspose(3, kernel_size=4, strides=2))
        model.add(LeakyReLU())

        model.summary()

        noise = Input(shape=(100,))
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))
        input = multiply([noise, label_embedding])

        img = model(input)  # output of the model

        return Model([noise, label], img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(32, kernel_size=3, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=3, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Conv2D(128, kernel_size=3, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Conv2D(256, kernel_size=3, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.summary()

        img = Input(shape=img_shape)

        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes + 1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, save_interval=50):
        # load data
        if self.dataset == 'mnist':
            (X_train, y_train), (_, _) = mnist.load_data()
            X_train = np.expand_dims(X_train, axis=3)
        elif self.dataset =='cifer':
            (X_train, y_train), (_, _) = cifar10.load_data()

        else:
            X_train, y_train = UTKFace_data(size=(self.img_rows, self.img_cols))

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        y_train = y_train.reshape(-1, 1)
        half_batch = int(batch_size // 2)

        # Class weights:
        # To balance the difference in occurences of digit class labels.
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        cw1 = {0: 1, 1: 1}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch
        class_weights = [cw1, cw2]

        for epoch in range(epochs):

            # --------------------
            # train Discriminator
            # --------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))
            # The labels of the digits that the generator tries to create and
            # image representation of
            sampled_labels = np.random.randint(0, 10, half_batch).reshape(-1, 1)

            sampled_labels = np.random.randint(0, self.num_classes, half_batch).reshape(-1, 1)
            gen_imgs = self.generator.predict([noise, sampled_labels])  # generated image

            fake_labels = self.num_classes * np.ones(half_batch).reshape(-1, 1)

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
            img_labels = y_train[idx]
            fake_labels = self.num_classes * np.ones(half_batch).reshape(-1, 1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels], class_weight=class_weights)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=class_weights)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -------------------
            # train Generator
            # -------------------
            noise = np.random.normal(0, 1, (batch_size, 100))

            valid = np.ones((batch_size, 1))
            # Generator wants discriminator to label the generated images as the intended
            # digits
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels],
                                                  class_weight=class_weights)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = self.num_classes//5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, r*c).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        fig.suptitle("ACGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].set_title("Class: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("acgan/images/"+self.dataset+"/output_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "acgan/saved_model/%s.json" % model_name
            weights_path = "acgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "mnist_acgan_generator")
        save(self.discriminator, "mnist_acgan_discriminator")
        save(self.combined, "mnist_acgan_adversarial")


if __name__ == '__main__':
    dcgan = ACGAN(32, 32, 3, 20, 'cifer')
    dcgan.train(epochs=6000, batch_size=32, save_interval=100)
