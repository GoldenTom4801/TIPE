
import random
import utility
import time


class GenMarkov:
    
    dic = {}
    texte = [] #Liste des mots du texte
    outputFile = None
    constructionTime = 0
    
    # Il faut fournir un texte, un fichier pour sauvegarder 
    def __init__(self,path, outputPath):
        self.outputPath = outputPath
        self.texte = utility.read_file(path)
        self.constructionDic()
        
    def __enter__(self):
        self.outputFile = open(self.outputPath, 'w')
        return self
    
    def __exit__(self,a,b,c):
        self.outputFile.close()
        
        
    #Comptage des mots
    def constructionDic(self):
        deb = time.time()
        for ind in range(len(self.texte) - 1):
            if self.texte[ind] in self.dic:
                self.dic[self.texte[ind]].append(self.texte[ind + 1]) 
            else:
                self.dic[self.texte[ind]] = [self.texte[ind + 1]]
        self.constructionTime = time.time() - deb
        print("Temps de construction : {0} sec".format(self.constructionTime))
        
    #Donne le prochain mot
    #Prend directement en compte les multiplicités
    #Note Attention à la complexité spatiale
    def prochainMot(self, mot):
        return random.choice(self.dic[mot])
    
            
    
    # On sélectionne un certain nombre de mots
    def genTexte(self,longueur, motInitial):
        motInitial = motInitial if motInitial in self.dic else self.texte[0]
        
        liste = [motInitial]
        for i in range(longueur -1):
            liste.append(self.prochainMot(liste[-1]))
        
        return " ".join(liste)
    

    def genPhraseBrute(self, motInitial):
        motInitial = motInitial if motInitial in self.dic else self.texte[0]
        
        liste = [motInitial]

        #Pour éviter de renvoyer directement si motInitial contient déjà un point
        #liste.append(self.prochainMot(liste[-1])) 

        while liste[-1][-1] != ".":  #Force à ce que la phrase se finisse par un point
            liste.append(self.prochainMot(liste[-1]))
            
        liste[0] = liste[0].capitalize()
        return liste

    def genPhrase(self, motInitial):
        liste = self.genPhraseBrute(motInitial)
        
        return " ".join(liste)
    

    
    def genParagraphe(self,motInitial, long = 0):
        long = 30 if long == 0 else long
            
        texte = ""
        for i in range(long-1):
            motsPhrase = self.genPhraseBrute(motInitial)
            motInitial = random.choice(self.dic[motsPhrase[-1]])
            texte += " ".join(motsPhrase) + "\n\n"
        return texte
    
    def writeParagraphe(self, motInitial, long):
        deb = time.time()
        self.outputFile.write(self.genParagraphe(motInitial, long) 
            + "\nDurée de génération :{0}".format(str(time.time()-deb)
            + "\nDurée de création du dictionnaire : {0}".format(self.constructionTime)))
                            
    def insight(self):
        #self.dic.
        print("Nombre moyen de successeurs : " )
        
        