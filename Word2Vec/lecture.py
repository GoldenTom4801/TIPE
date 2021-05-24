
#Methode pour l'extraction de mots 

def lire_fichier(entree):
    fichier = open(entree)
    texte = fichier.read().lower().split() #minuscule
    return texte
# Renvoie le texte en entier, en réduisant en minuscule 

def creer_correspondance(texte):
    mots_ind = dict()
    ind_mots = dict()
    unique = set(texte)
    for i,mot in enumerate(unique):
        mots_ind[mot] = i
        ind_mots[i] = mot

    return mots_ind, ind_mots
#Transforme les mots en indices 
    
    
    
