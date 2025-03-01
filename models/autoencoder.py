import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, \
    concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class Autoencoder:
    def __init__(self, input_shape=(256, 256, 3), latent_dim=128, model_path='models/saved/autoencoder'):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model_path = model_path
        self.model = None
        self.threshold = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(self.latent_dim, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)

        outputs = Conv2D(self.input_shape[-1], (3, 3), padding='same', activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

        self.model = model
        return model

    def train(self, normal_images, validation_split=0.2, epochs=100, batch_size=32):
        if self.model is None:
            self.build_model()

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        callbacks = [
            ModelCheckpoint(f"{self.model_path}.h5", save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]

        history = self.model.fit(
            normal_images, normal_images,
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

    def predict(self, images):
        if self.model is None:
            raise ValueError("Model not built or loaded")

        return self.model.predict(images)

    def compute_anomaly_scores(self, images):
        reconstructions = self.predict(images)

        mse = np.mean(np.square(images - reconstructions), axis=(1, 2, 3))

        error_images = np.mean(np.square(images - reconstructions), axis=-1)

        return mse, error_images, reconstructions

    def set_threshold(self, normal_images, percentile=95):
        mse, _, _ = self.compute_anomaly_scores(normal_images)
        self.threshold = np.percentile(mse, percentile)
        return self.threshold

    def detect_anomalies(self, images, pixel_threshold=None):
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold first.")

        mse, error_images, reconstructions = self.compute_anomaly_scores(images)

        is_anomaly = mse > self.threshold

        if pixel_threshold is None:
            pixel_threshold = np.mean(error_images) + 2 * np.std(error_images)

        anomaly_masks = error_images > pixel_threshold

        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': mse,
            'reconstruction': reconstructions,
            'error_images': error_images,
            'anomaly_masks': anomaly_masks
        }