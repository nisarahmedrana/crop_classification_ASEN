import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dot, Softmax, Lambda

class ASEN(Model):
    def __init__(self, num_learners, num_classes):
        super(ASEN, self).__init__()
        self.hidden = Dense(64, activation='relu')
        self.attention_vector = self.add_weight(shape=(64, 1), initializer='random_normal', trainable=True)
        self.softmax = Softmax(axis=1)
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        h = self.hidden(inputs)  # shape: (batch_size, num_learners, 64)
        attention_scores = tf.squeeze(tf.matmul(h, self.attention_vector), axis=-1)
        attention_weights = self.softmax(attention_scores)
        weighted_inputs = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return self.output_layer(weighted_inputs)