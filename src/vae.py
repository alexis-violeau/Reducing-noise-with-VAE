import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a curve."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    """Variational autoencoder that learn to reconstruct denoised curves."""    
    def __init__(self, input_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        
        ## Constructing encoder from input space to latent space
        encoder_inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(128, activation="relu",name = 'hidden_layer_encoder',kernel_regularizer = 'l2')(encoder_inputs)
        
        z_mean = layers.Dense(2, activation = 'linear', name="z_mean")(x)
        z_log_var = layers.Dense(2, activation = 'linear', name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        
        ## Constructing decoder from latent space to input space
        latent_inputs = keras.Input(shape=(2,))
        x = layers.Dense(128, activation="relu", name = 'hidden_layer_decoder',kernel_regularizer = 'l2')(latent_inputs)

        decoder_outputs = layers.Dense(input_dim, activation="linear")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")        
        

    @property
    
        
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """Custom train step for VAE"""

        with tf.GradientTape() as tape:
            # First, we compute the forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Then we compute the different losses
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(data, reconstruction)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        # We compute the gradient and backpropage the gradient (backward) 
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # We update the loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
