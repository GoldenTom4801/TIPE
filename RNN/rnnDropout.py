# Création d'un réseau neuronal récurrent (RNN) dont le but est de générer caractère par caractère un texte à partir d'un
# texte d'entrainement 

import tensorflow as tf
import numpy as np
import os
import json
import utility

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

seq_long = 100
batch_per_epochs = len(text) // (seq_long + 1) # Chaque batch fera 100 caractères (et on jette 1 caractère)

sequences = ids_dataset.batch(seq_long + 1, drop_remainder = True) 
    # On enlève le dernier batch incomplet 

def split_input_target(seq):
    input = seq[:-1] # On enlève le dernier caractère
    target = seq[1:] # On enlève le premeir caractère
    return (input, target)


dataset = sequences.map(split_input_target) 
# les données finalement utilisables de la forme [([char_input], [char_target])] 

#Création des batchs d'entrainement
    #Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE) # Pas sûr de comprendre
    .batch(BATCH_SIZE, drop_remainder = True)
    .prefetch(tf.data.experimental.AUTOTUNE) # C'est quoi ??
) #Effet de bord ou non ? Forcément non mais c'est bizarre

# Création du modèle

    # Nombre de caractère unique (taille de vocabulaire), donc de la couche de sortie !
vocab_size = len(vocab)
    # La taille des vecteurs de représentations
dim_vect = 256 # C'est beaucoup non ?
    # Nombre de neuronnes RNN (couche cachée)
rnn_units =  1024

model = utility.RNNGenTexteDropout(
    vocab_size= len(ids_from_chars.get_vocabulary()),
    dim_vect = dim_vect,
    rnn_units = rnn_units,
    dropProp=0.5
)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer = "adam", loss = loss)
"""
# Petit test pour voir si ça marche ^^. Note : Il semble y avoir un problème pour obtenir sous forme de string
# à cause de la fonction text_from_ids
example_batch_predictions = None
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
samples_indices = tf.random.categorical(example_batch_predictions[0], num_samples = 1)

example_batch_predictions = model(input_example_batch)
samples_indices = tf.squeeze(samples_indices, axis = -1).numpy()

examples_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = examples_batch_loss.numpy().mean() 

print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss) """

# Crée le callback (afin de sauvegarder à chaque epoch)
checkpoint_dir = "training_checkpoints/"
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.ckpt")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
) 


history = model.fit(dataset, epochs = 50, callbacks = [checkpoint_callback])
history = history.history

json.dump(history, open("RNN/training_checkpoints/history.json", 'w')) # Attention à changer la localisation 
#Cela permet notamment d'accéder à l'historique des loss !!

#model.save_weights("RNN/model_trained/localTest")

#path = "RNN/model_trained/Fontaine1/modèle"
#model = tf.keras.models.load_model(path)


