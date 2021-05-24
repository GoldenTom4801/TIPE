
import random
import utility
import time
import numpy as np
import matplotlib.pyplot as plt


class GenMarkov:
    
    dic = {} #Associe à un mot la liste de ses successeurs avec multiplicité
    texte = [] #Liste des mots du texte
    outputFile = None # Pour sauvegarder les phrases
    constructionTime = 0 #On conserve le temps de construction du dic
    
    # Il faut fournir un texte, un fichier pour sauvegarder 
    def __init__(self,path, outputPath):
        self.outputPath = outputPath
        self.texte = utility.read_file(path)
        print("hey")
        self.constructionDic()
        
        
        
    # Permet d'utiliser avec "with" pour fermer automatiquement le fichier
    def __enter__(self):
        self.outputFile = open(self.outputPath, 'w')
        return self
    
    def __exit__(self,a,b,c):
        self.outputFile.close()
        
        
    #Comptage des mots
        #Présente directement le temps de construction pour des mesures de 
        #performances

    def constructionDic(self):
        deb = time.time()
        for ind in range(len(self.texte) - 1):
            if self.texte[ind] in self.dic:
                if self.texte[ind+1] in self.dic[self.texte[ind]]:
                    self.dic[self.texte[ind]][self.texte[ind + 1]] += 1
                else:
                    self.dic[self.texte[ind]][self.texte[ind + 1]] = 1
            else:
                nDic = dict()
                nDic[self.texte[ind + 1]] = 1
                self.dic[self.texte[ind]] = nDic

        taille = 30
        somme = 0
        repetitionSuccesseur = np.zeros(taille)
        outOfRange = []
        for mot in self.dic:
            for motSucc, value in self.dic[mot].items():
                if value>= 2:
                    if value<taille:
                        repetitionSuccesseur[value] = repetitionSuccesseur[value] + 1
                    else:
                        outOfRange.append((mot, motSucc, value))
            self.dic[mot] = ([x for x in self.dic[mot]], [self.dic[mot][x] for x in self.dic[mot]])
            somme += len(self.dic[mot][0]) # Nombre des successeurs distincts
        for obj in outOfRange:
            print(obj)
        #print(outOfRange)
        # plt.plot(np.arange(taille), repetitionSuccesseur)
        # plt.show()
        print(somme, somme/len(self.texte))
        
        

        self.constructionTime = time.time() - deb
        print("Temps de construction : {0} sec".format(self.constructionTime))  
    #On parcourt le texte
    # On crée un dictionnaire de dictionnaire (ajout d'élément de complexité élémentaire)
    # On transforme la structure en un dictionnaire qui à un élément associe la liste des successeurs
    # et la liste des probabilités associées 
        
        
    #Donne le prochain mot
    #Prend directement en compte les multiplicités
    #Note : Attention à la complexité spatiale
    # On pourrait changer et conserver le nombre représentant la multiplicité
    def prochainMot(self, mot):
        #print(self.dic[mot])
        successeur, poids = self.dic[mot]

        return random.choices(successeur, weights=poids)[0] #Renvoie en fait une liste
    
            
    
    # On sélectionne un certain nombre de mots et on choisit
    # successivement le prochain mot
    # Note : ne renvoie pas forcément une phrase (qui se finit par un point)
    def genTexte(self,longueur, motInitial):
        motInitial = motInitial if motInitial in self.dic else self.texte[0] 
            #Assure que le mot est bien présent dans le dictionnaire
        
        liste = [motInitial]
        for i in range(longueur -1):
            liste.append(self.prochainMot(liste[-1]))
        
        return " ".join(liste)
    
    # Permet des manipulations sur la phrase
    # En fait, useless car il suffit de faire un split
    def genPhraseBrute(self, motInitial):
        motInitial = motInitial if motInitial in self.dic else self.texte[0]
        
        liste = [motInitial]
        #Rip si le mot initial contient un point à la fin

        while liste[-1][-1] != ".":  #Force à ce que la phrase se finisse par un point
            liste.append(self.prochainMot(liste[-1]))
            
        liste[0] = liste[0].capitalize()
        return liste

    #On renvoie bien la phrase en str
    def genPhrase(self, motInitial):
        liste = self.genPhraseBrute(motInitial)
        
        return " ".join(liste)
    

    #Ecrit long nombre de phrases.
    # Le premier mot est motInitial
    # Le premier mot des autres phrases est un des successeurs du dernier
    # mot de la phrase précédente (bref les phrases s'enchainent)
    def genParagraphe(self,motInitial, long):
            
        texte = ""
        for i in range(long-1):
            motsPhrase = self.genPhraseBrute(motInitial)
            successeur, poids = self.dic[motsPhrase[-1]]
            motInitial = random.choices(successeur,weights=poids)[0]
            texte += " ".join(motsPhrase) + "\n\n"
        return texte
    
    
    #Ecrit directement dans le fichier output (de chemin path)
    def writeParagraphe(self, motInitial, long):
        deb = time.time()
        self.outputFile.write(self.genParagraphe(motInitial, long) 
            + "\nDurée de génération :{0}".format(str(time.time()-deb)
            + "\nDurée de création du dictionnaire : {0}".format(self.constructionTime)))
                            
                            
    # Beh renvoie le nombre moyen de successeur 
    def nbMoyenSuccesseur(self, phrase): 
        mots = phrase.split()

        nbMoyen = utility.fold_left((lambda a,b: a + len(self.dic[b.lower()])), 0, mots)/len(mots)
        return nbMoyen
    # Note : la mesure n'est pas toujours la plus pertinente
    # Car les petits mots comme "de" sont de contributions trop grande
    
        
    # Pour l'instant, la mesure consiste à savoir si un triplet de 3 mots est 
    # présent dans le texte
    def mesurePhrase(self, phrase):
        mots = phrase.split()
        
        triplet = [] 
        for i in range(len(mots)-2):
            triplet.append(" ".join(mots[i:i+3]))
        
        for morceau in triplet:
            if morceau in self.texte:
                return (True, morceau) # On renvoie le fautif
        return (False, None) #Pour avoir le même type de sortie 
        # C'est un peu violent 

        
        
        