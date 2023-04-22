import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

class MobileNetV2():
    def __init__(self,dataset=None, classNames=None):
        # self.model was formerly base_model
        BATCH_SIZE = 32
        self.IMG_SIZE = (160, 160)
        
        if dataset:
            PATH = dataset
            self.train_dir = os.path.join(PATH, 'train')
            self.validation_dir = os.path.join(PATH, 'validation')
            self.train_dataset = tf.keras.utils.image_dataset_from_directory(self.train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=self.IMG_SIZE)
            self.validation_dataset = tf.keras.utils.image_dataset_from_directory(self.validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=self.IMG_SIZE)
            self.class_names = self.train_dataset.class_names
        
        else:

            _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
            path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
            PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

            

            self.train_dir = os.path.join(PATH, 'train')
            self.validation_dir = os.path.join(PATH, 'validation')
            self.train_dataset = tf.keras.utils.image_dataset_from_directory(self.train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=self.IMG_SIZE)
            self.validation_dataset = tf.keras.utils.image_dataset_from_directory(self.validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=self.IMG_SIZE)

            AUTOTUNE = tf.data.AUTOTUNE
            self.class_names = self.train_dataset.class_names
            print(self.train_dataset)

        
        self.base_learning_rate = 0.0001

    
    def show_train(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")

    def get_batches(self):
        val_batches = tf.data.experimental.cardinality(self.validation_dataset)
        self.test_dataset = self.validation_dataset.take(val_batches // 5)
        self.validation_dataset = self.validation_dataset.skip(val_batches // 5)
        print('Number of validation batches: %d' % tf.data.experimental.cardinality(self.validation_dataset))
        print('Number of test batches: %d' % tf.data.experimental.cardinality(self.test_dataset))
   
    def augment(self):
        self.data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),tf.keras.layers.RandomRotation(0.2),])
        for image, _ in self.train_dataset.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = self.data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis('off')

    def pre_process(self):
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

        # Create the base model from the pre-trained model MobileNet V2
        IMG_SHAPE = self.IMG_SIZE + (3,)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

        image_batch, label_batch = next(iter(self.train_dataset))
        feature_batch = self.base_model(image_batch)
        print(feature_batch.shape)

        self.base_model.trainable = False

        # Let's take a look at the base model architecture
        self.base_model.summary()

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print(feature_batch_average.shape)

        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)
        print(prediction_batch.shape)


        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = self.data_augmentation(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)

        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self.model.summary()

        len(self.model.trainable_variables)

        self.initial_epochs = 10

        loss0, accuracy0 = self.model.evaluate(self.validation_dataset)

        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        self.history = self.model.fit(self.train_dataset,
                    epochs=self.initial_epochs,
                    validation_data=self.validation_dataset)
        
    def accuracy_and_loss(self):
        self.acc = self.history.history['accuracy']
        self.val_acc = self.history.history['val_accuracy']

        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.acc, label='Training Accuracy')
        plt.plot(self.val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(self.loss, label='Training Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

    def fine_tuning(self):
        self.base_model.trainable = True

        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(self.base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 100

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False


        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.base_learning_rate/10),
                    metrics=['accuracy'])

        self.model.summary()

        len(self.model.trainable_variables)

        self.fine_tune_epochs = 10
        total_epochs =  self.initial_epochs + self.fine_tune_epochs

        self.history_fine = self.model.fit(self.train_dataset,
                                epochs=total_epochs,
                                initial_epoch=self.history.epoch[-1],
                                validation_data=self.validation_dataset)
        
    def fine_accuracy_and_loss(self):
        self.acc += self.history_fine.history['accuracy']
        self.val_acc += self.history_fine.history['val_accuracy']

        self.loss += self.history_fine.history['loss']
        self.val_loss += self.history_fine.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.acc, label='Training Accuracy')
        plt.plot(self.val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([self.initial_epochs-1,self.initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(self.loss, label='Training Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([self.initial_epochs-1,self.initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

        self.loss, accuracy = self.model.evaluate(self.test_dataset)
        print('Test accuracy :', accuracy)



    def show_predictions(self):
        # Retrieve a batch of images from the test set
        image_batch, label_batch = self.test_dataset.as_numpy_iterator().next()
        predictions = self.model.predict_on_batch(image_batch).flatten()

        # Apply a sigmoid since our model returns logits
        # issue seems to be here, 3 classes as opposed to to two with the cats and dogs example.
        # look for a multi-class example to compare against not sure how to get predictions but all Dermacentor Variabilis is wrong
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)

        print('Predictions:\n', predictions.numpy())
        print('Labels:\n', label_batch)

        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].astype("uint8"))
            plt.title(self.class_names[predictions[i]])
            plt.axis("off")

    def save_model(self, path):
        self.model.save(path)
        return