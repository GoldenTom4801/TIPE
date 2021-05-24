import tensorflow as tf
import numpy as np
import os
import json

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

# Traitement du texte (pour être utilisable par le RNN)
# A chaque étape, on veut trouver le prochain caractère
    # Il suffit d'avoir un segment de caractères (input) et le même décalé de 1 caractère à droite (target)


text_ids = ids_from_chars(splitText(text))
ids_dataset = tf.data.Dataset.from_tensor_slices(text_ids)


sequences = ids_dataset.batch(2, drop_remainder = False)

def split_input_target(seq): # A partir d'une lettre avoir la suivante
    return seq[:1], seq[1:]

dataset = sequences.map(split_input_target)


#Création des batchs d'entrainement
    #Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000
nb_epochs = 80

dataset = (
     dataset
     .shuffle(BUFFER_SIZE)
     .batch(BATCH_SIZE, drop_remainder = True)
     .prefetch(tf.data.experimental.AUTOTUNE)
 )

dim_vect = 128

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(ids_from_chars.get_vocabulary()), dim_vect),
    tf.keras.layers.Dense(256, activation = "relu"),
    tf.keras.layers.Dense(len(ids_from_chars.get_vocabulary()))
]
)
# Le callback pour enregistrer automatiquement
checkpoint_dir = "drive/MyDrive/training_checkpointsNaif/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.ckpt")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    period = 10
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer = "adam", loss = loss)
model.fit(dataset, epochs = nb_epochs, callbacks =[checkpoint_callback])