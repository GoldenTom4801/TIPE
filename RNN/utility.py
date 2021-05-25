# On pourra rajouter des fonctions utiles partout
import tensorflow as tf
import numpy as np
import os

class RNNGenTexte(tf.keras.Model): # Notre classe pour RNN
    def __init__(self, vocab_size, dim_vect, rnn_units):
        super().__init__(self)
        
        self.embedding = tf.keras.layers.Embedding(vocab_size,dim_vect)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                        return_sequences=True,
                                        return_state= True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states = None, return_state= False, training=False):
        x = inputs 
        x = self.embedding(x, training = training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state = states, training = training)
        x = self.dense(x, training = training)

        if return_state:
            return x, states
        else:
            return x

class RNNGenTexteDropout(tf.keras.Model): # Notre classe pour RNN
    def __init__(self, vocab_size, dim_vect, rnn_units, dropProp):
        super().__init__(self)
        
        self.embedding = tf.keras.layers.Embedding(vocab_size,dim_vect)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                        return_sequences=True,
                                        return_state= True)
        self.dropout = tf.keras.layers.Dropout(dropProp)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states = None, return_state= False, training=False):
        x = inputs 
        x = self.embedding(x, training = training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state = states, training = training)
        x = self.dropout(x, training = training)
        x = self.dense(x, training = training)

        if return_state:
            return x, states
        else:
            return x

class DoubleRNNGenTexte(tf.keras.Model): # Notre classe pour RNN
    def __init__(self, vocab_size, dim_vect, rnn_units, dropProp):
        super().__init__(self)
        
        self.embedding = tf.keras.layers.Embedding(vocab_size,dim_vect)
        self.gru1 = tf.keras.layers.GRU(rnn_units,
                                        return_sequences=True,
                                        return_state= True)
        self.dropout = tf.keras.layers.Dropout(dropProp)
        self.gru2 = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states = None, return_state= False, training=False):
        x = inputs 
        x = self.embedding(x, training = training)
        if states is None:
            states = self.gru1.get_initial_state(x)
        x, states = self.gru1(x, initial_state = states, training = training)
        x = self.dropout(x)
        x, states = self.gru2(x, initial_state=states, training = training)
        x = self.dense(x, training = training)

        if return_state:
            return x, states
        else:
            return x

class RNNGenTexteOneHot(tf.keras.Model): # Notre classe pour RNN
    def __init__(self, vocab_size, rnn_units):
        super().__init__(self)
        # On lui envoie directement des vecteurs one_hot
        self.gru = tf.keras.layers.GRU(rnn_units,
                                        return_sequences=True,
                                        return_state= True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states = None, return_state= False, training=False):
        x = inputs 
        x = self.embedding(x, training = training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state = states, training = training)
        x = self.dense(x, training = training)

        if return_state:
            return x, states
        else:
            return x