#Fonctions qui servent pour le reste du projet

def read_file(chemin):
    with open(chemin, 'r') as f:
        return f.read().split("\n")
    
def concat(*args):
    texte = ""
    sans_vide = del_vide(args)
    for i in range(len(sans_vide)-1):
        texte += sans_vide[i] + " "
        
    return texte + sans_vide[-1]
    
def del_vide(liste):
    new_liste = []
    for i in liste:
        if i != "":
            new_liste.append(i)
    return new_liste
