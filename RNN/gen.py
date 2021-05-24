import time

# Création d'un réseau neuronal récurrent (RNN) dont le but est de générer caractère par caractère un texte à partir d'un
# texte d'entrainement 

import tensorflow as tf
import numpy as np
import os
from utility import RNNGenTexte

#Ouverture du fichier texte (d'entrainement)
text_path = "RNN/sample_data/fables_fontaine.txt" # Attention, on part de la base du projet
text = ""
with open(text_path, "r") as f:
    text = f.read()


#vocabulaire (caractères uniques)
vocab = sorted(set(text))


# Vectorisation - Fonctions pour le reste de l'algorithme
ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary= ids_from_chars.get_vocabulary(), 
                                                                    invert = True,
                                                                    mask_token=None)

def splitText(liste): # A utiliser avant d'envoyer du texte à ids_from_chars
    return tf.strings.unicode_split(liste, input_encoding = "UTF-8")

def text_from_ids(ids): # Il faudrait que je sache vraiment pourquoi axis = -1 (axis = 1 semble marche aussi, pas = 0)
    return tf.strings.reduce_join(chars_from_ids(ids), axis = 1)

#vocab_size = len(vocab) #Inutile car il faut prendre en compte les caractères inexistants
    # La taille des vecteurs de représentations
dim_vect = 256 # C'est beaucoup non ?
    # Nombre de neuronnes RNN (couche cachée)
rnn_units =  1024

model = RNNGenTexte(len(ids_from_chars.get_vocabulary()), dim_vect, rnn_units)
model.build((1, 100))
model.load_weights("RNN/model_trained/training/ckpt_100.ckpt")


class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
start = time.time()
states = None
next_char = tf.constant(['FABLE:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)