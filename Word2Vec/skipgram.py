from lecture import *
import numpy as np
import random
from matplotlib import pyplot as plt

#Skip Gram # On va vraiment faire un skigram
#On essaie de trouver le contexte en fonction du mot présent

#Attention dans tout le programme les vecteurs mots sont des vecteurs lignes
# De même la matrice de représentation des mots est la concaténation des lignes des mots

TAILLE_CONTEXTE = 4 #Fenêtre d'entrainement
TAILLE_VECTEUR = 128 #Dimension du vecteur représentatif des mots
ALPHA = 0.01 #Facteur d'apprentissage. Choisi ici au hasard pour l'instant 

poids_entree = None
poids_cachee = None

output = "loss.txt"

texte = lire_fichier("input/fables_fontaine.txt") #On récupère les données à partir de "lecture"
taille_vocab = len(set(texte))
mots_ind, ind_mots = creer_correspondance(texte)

dossier = "output/"

def enregistrer_donnees(name): #Enregistre les matrices
    np.savetxt(dossier+'data_poids_c'+ name+ '.csv', poids_cachee, delimiter=',')
    np.savetxt(dossier+'data_poids_e' +name+ '.csv', poids_entree, delimiter=',')

def lecture_donnes(name): #Lecture des matrices entrainnées
    #Il faudra changer à la main la localisation des fichiers
    global poids_entree
    global poids_cachee
    poids_cachee = np.loadtxt(dossier+'data_poids_c'+name +'.csv', delimiter=',')
    poids_entree = np.loadtxt(dossier+'data_poids_e'+name+'.csv', delimiter=',')


def donnees_entrainement(texte, indice):
    liste = [mots_ind[texte[indice+i]] for i in range(-TAILLE_CONTEXTE, 0)]
    liste[-1:] = [mots_ind[texte[indice+i]] for i in range(1, TAILLE_CONTEXTE+1)]
    return liste
#Récupère les mots de contextes dans la fenêtre d'entrainement du texte original

def init_couche_entree():
    global poids_entree
    poids_entree = np.random.randn(taille_vocab,TAILLE_VECTEUR) * 0.1
def init_couche_cachee():
    global poids_cachee
    poids_cachee = np.random.randn(TAILLE_VECTEUR, taille_vocab) * 0.1
#Les deux fonctions permettent d'initialiser de façon aléatoire les matrices de représentation et la couche cachée

def one_hot_vector(mot_entree, mots_contextes): #Note, les variables d'entrees sont bien des indices
    vec_input = np.zeros(taille_vocab) # Crée un vecteur nul

    vec_input[mot_entree] = 1

    vec_contexte = np.zeros(taille_vocab) #idem
    for indice in mots_contextes:
        vec_contexte[indice] = 1
    return vec_input, vec_contexte
    


def ind_vecteurs_mots(indice): #Extrai la colonne, coute moins cher qu'une multiplication
    one_vect, _ = one_hot_vector(indice, [])
    vect_mots = np.dot(poids_entree.T,one_vect) # On pourra optimier
    return vect_mots
#Permet de former la matrice dont les colonnes sont les différents mots dans leur
#représentation vectorielle
# C'est la couche d'entree

def couche_cachee(vecteur_mots):
    vec_sortie = np.dot(poids_cachee.T, vecteur_mots)
    return vec_sortie
#Passage dans la couche cachée,
#la sortie est celle qui est ensuite passée à la softmax

def softmax_vec(Z):
    if np.sum(np.exp(Z)) == 0:
        print(Z)
    #renvoie le vecteur auquel on a appliqué un softmax
    valeur = np.divide(np.exp(Z), np.sum(np.exp(Z)))
    return valeur

def softmax(valeur):
    return 1 / (1 + np.exp(-1*valeur))


def propagation_avant(indice, mots_contexte): #mots_contexte est inutile ici
    vect_mot = ind_vecteurs_mots(indice)
    #print(vect_mot)
    vect = couche_cachee(vect_mot)
    #print("l'autre vecteur")
    #print(vect)
    softmax_vect = softmax_vec(vect)

    return softmax_vect, vect, vect_mot
# C'est la fonction qui fait toute l'étape forward !
# Renvoie le vecteur de probabilité

def calculate_error(y_pred,context_words):
    
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}
    
    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update ( {index : 'yes'} )
        
    number_of_1_in_context_vector = len(index_of_1_in_context_words)
    
    for i,value in enumerate(y_pred):
        
        if index_of_1_in_context_words.get(i) != None:
            total_error[i]= (value-1) + ( (number_of_1_in_context_vector -1) * value)
        else:
            total_error[i]= (number_of_1_in_context_vector * value)
            
            
    return  np.array(total_error)

def update_poids(indice, mots_contexte):
    global poids_cachee
    global poids_entree
    
    softmax_vect, vec, vect_mot = propagation_avant(indice, mots_contexte)
    #print(vec)
    #e = softmax_vect*len(mots_contexte) - mots_contexte
    e = calculate_error(softmax_vect, mots_contexte)

    entree, _ = one_hot_vector(indice, [])
    
    derivee_cachee = np.outer(vect_mot, e)
    derivee_entree = np.outer(entree, np.dot(poids_cachee, e))

    #print(derivee_cachee)
    #print(poids_cachee)
    #plt.matshow(poids_cachee - ALPHA * derivee_cachee)
    #plt.matshow(poids_cachee)
    
    poids_cachee = poids_cachee - ALPHA * derivee_cachee
    poids_entree = poids_entree - ALPHA * derivee_entree
    return calculer_perte(softmax_vect, mots_contexte)
                              

def calculer_perte(u, contexte):
    somme_1 = 0
    for index in np.where(contexte==1)[0]:
        somme_1 -= u[index]

    somme_2 = len(np.where(contexte==1)[0]) * np.log(np.sum(np.exp(u)))
    return somme_1 + somme_2



"""def update_valeur(alpha, matrice_mot, matrice_target):
    global matrice_representation
    taille = np.shape(matrice_mot)[0]
    for i in range(taille):
        new_vect = matrice_mots[i] - alpha*derivee_mot_contexte(matrice_mot[i], matrice_target[i])
        matrice_representation[ind_mot[matrice_target = """
#Applique la descente de gradient sur la matrice des mots
# Pour l'instant du contexte d'un mot

def chercher_indice(mot):
    indice = []
    for ind, unMot in enumerate(texte):
        if unMot == mot:
            indice.append(ind)
    return indice

def entrainement(nb_epochs):
    dictio = dict()
    for mot in texte:
        dictio[mot] = ([], [])
    loss_value = []
    coord = []
    for i in range(nb_epochs):
        loss = 0
        for indice in range(TAILLE_CONTEXTE, len(texte) - TAILLE_CONTEXTE):
            #indice = random.randint(TAILLE_CONTEXTE, len(texte) - TAILLE_CONTEXTE-1)
            #indice = 9
            indice_contexte = donnees_entrainement(texte, indice)
            
            vect_input, vect_contexte = one_hot_vector(mots_ind[texte[indice]], indice_contexte)
            if indice % 100 == 0:
                print(str(indice/len(texte)/100) + "%")
            
            
            loss += update_poids(mots_ind[texte[indice]], vect_contexte)
            #print(loss)
        if i % 1 == 0:
            print("epoch " + str(i) + ", coût :" + str(loss))
            enregistrer_donnees("fontaine" + str(i))
            
            #coord.append(poids_entree[0,0])
            #cout.append(loss)
            """cout, ind = dictio[texte[indice]]
            cout.append(loss)
            ind.append(i)"""
        loss_value.append(loss)
        
    with open(dossier + output, "w") as f:
        f.write(str(loss_value))
        
    """donnes = dictio.items()
    for key, (cout, indice) in donnes:
        plt.plot(indice, cout, "+")
        #plt.show()"""
    plt.plot(range(nb_epochs), loss_value)
    plt.show()

def generer_texte(first_word, longueur): # La fonction qui permettra ensuite de créer un texte
    
    
    
    texte = []
    mot = first_word
    texte.append(mot)
    for i in range(longueur):
        softmax_value, _,_ = propagation_avant(mots_ind[mot], [])
        print(softmax_value)
        choix = random.choices(range(taille_vocab), weights=softmax_value)
        mot = ind_mots[choix[0]] #Choix est une liste à 1 élément étrangement
        texte.append(mot)
    return " ".join(texte)
    
 
init_couche_cachee()
init_couche_entree()

#lecture_donnes("fable_fontaine_50")

entrainement(50)

enregistrer_donnees("fable_fontaine_50")



#print(generer_texte("les",50)) #Le programme est actuellement (un peu) entrainé, faire des testes de générations de phrases

#print(matrice_representation)
#print(ind_vecteurs_mots(vecteur))



    

    
    

