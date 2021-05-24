
# Fonctions pour le reste (à copier/copier dans 

def fold_left(f, a, liste):
    for b in liste:
        a = f(a, b)
    return a


# --

import json
import urllib.request

def getNature(mot):
    # Donnée pour compléter l'url afin de faire la recherche
    nbSource = 1 
        # Un mot peut avoir plusieurs classes, mais on va choisir le premier
        # Pour l'instant, aucun moyen de savoir si c'est la bonne
    KEY = "Hm8SN0GSxxsaEKy2BZ3hJ3f04YfBuBnu"
    urlVide = "https://api.dicolink.com/v1/mot/{0}/definitions?limite={1}&api_key={2}"

    

    urlComplete = urlVide.format(mot, nbSource,KEY)
    req = urllib.request.Request(urlComplete, headers={'User-Agent': 'Mozilla/5.0' })
        # Bien que l'outils soit fait pour être utilisé avec des algos
        # Le serveur rejette les demandes lorsqu'elles n'ont pas de headers (protection
        # contre le spam). On lui en fournit un arbitraire 
        
    
    jsonDonnee = urllib.request.urlopen(req)
    
    data = json.load(jsonDonnee)[0] # On extrait une liste de json, on récupère le premier élément
    return data["nature"]
    
# --



