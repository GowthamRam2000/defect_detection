import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class UNetModel:
    def __init__(self, input_shape=(256, 256, 3), model_path='models/saved/unet'):
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None

    def conv_block(self, inputs, filters, kernel_size=(3, 3), padding='same', dropout_rate=0.1):
        x = Conv2D(filters, kernel_size, padding=padding, activation='relu')(inputs)
        x = Conv2D(filters, kernel_size, padding=padding, activation='relu')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        return x

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Encoder
        conv1 = self.conv_block(inputs, 64)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_block(pool1, 128)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_block(pool2, 256)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_block(pool3, 512)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bridge
        conv5 = self.conv_block(pool4, 1024)

        # Decoder
        up6 = Conv2D(512, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv5))
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = self.conv_block(merge6, 512)

        up7 = Conv2D(256, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = self.conv_block(merge7, 256)

        up8 = Conv2D(128, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = self.conv_block(merge8, 128)

        up9 = Conv2D(64, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = self.conv_block(merge9, 64, dropout_rate=0)

        # Output
        outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(conv9)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])])

        self.model = model
        return model

    def train(self, images, masks, validation_split=0.2, epochs=100, batch_size=16):
        if self.model is None:
            self.build_model()

        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Define callbacks
        callbacks = [
            ModelCheckpoint(f"{self.model_path}.h5", save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]

        # Train the model
        history = self.model.fit(
            images, masks,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True
        )

        return history

    def load_weights(self, weights_path=None):
        if weights_path is None:
            weights_path = f"{self.model_path}.h5"

        if self.model is None:
            self.build_model()

        self.model.load_weights(weights_path)

    def predict(self, images, threshold=0.5):
        if self.model is None:
            raise ValueError("Model not built or loaded")

        # Get raw predictions (probabilities)
        pred_masks = self.model.predict(images)

        # Apply threshold to get binary masks
        binary_masks = (pred_masks > threshold).astype(np.float32)

        return pred_masks, binary_masks

    def evaluate(self, images, true_masks, threshold=0.5):
        pred_masks, binary_masks = self.predict(images, threshold)

        # Evaluate using model's metrics
        metrics = self.model.evaluate(images, true_masks)

        # Get model's metrics names
        metrics_names = self.model.metrics_names

        # Create evaluation dictionary
        evaluation = dict(zip(metrics_names, metrics))

        return {
            'evaluation': evaluation,
            'predictions': pred_masks,
            'binary_masks': binary_masks
        }