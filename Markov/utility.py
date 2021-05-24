
import re

# Fonctions utiles dans tout le projet 


def fold_left(f, a, liste):
    for b in liste:
        a = f(a, b)
    return a

def fold_right(f, liste, b):
    for ind in range(len(liste), -1, -1):
        b = f(liste[ind], b)
    return b



# Renvoie le mot en minuscule
# Ecrit de cette façon pour avoir une fonction et non une méthode
def minuscule(mot):
    return mot.lower()

# A partir d'un str représentant un texte renvoie la liste des mots
# (On cherche à enlever les espaces " " et les sauts de ligne "\n")
# Note : on pourrait faire un fold_left
def brutVersListe(texte):
    liste = texte.split("\n")
    liste = list(map(lambda phrase : phrase.split(), liste))
    listeMot = []
    for listePhrase in liste:
        listeMot += listePhrase
    return listeMot
        
# On effectue un traitement supplémentaire en mettant tous les mots en minuscule
# Note : c'est un choix dans la méthode 
def read_file(chemin):
    with open(chemin, 'r') as f:
        listeMots =  brutVersListe(f.read())
        return list(map(minuscule, listeMots))
    