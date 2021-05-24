import utility
import random
from _operator import concat

class Generateur(object):
    
    # Dossier dans lequel se trouve les fichiers de mots
    dossier = "Chomsky/data"
    
    # Listes des mots en fonctions de leur classe 
    preps = []
    advs = []
    adjs = []
    articles = []
    noms = []
    verbes = []
    
    # Il vaudrait mieux lui donner le dossier
    def __init__(self, output_file, preps = [], advs = [], adjs = [], articles = [], noms = [], verbes = []):
        # Ouvre le fichier pour les sauvegardes selon le chemin (output_file) fourni
        self.output_file = open(output_file, 'w')
            # Plutôt à mettre dans __enter__ normalement
            
            
        # Soit on lui fournit les listes soit il va les chercher (on suppose tout ou rien)
        if preps == []:
            self.readData()
        else:
            self.preps = preps
            self.advs = advs
            self.adjs = adjs
            self.articles = articles
            self.noms = noms
            self.verbes = verbes
            
        
        

    # Pour l'utilisation avec "with"
    def __enter__(self):
        # Il faudrait plutôt appeler ici une fonction pour ouvrir le fichier
        return self
        # Note : c'est d'ailleurs un peu bête car je ne fournis pas de 
        # façon de décharger le fichier si je ne passe pas par un "with"
    
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.output_file.close()
            # On referme le fichier (et libère ainsi la mémoire)
            
    def readData(self):
            # Chemins des fichiers
        # Les noms sont choisis de façon standard, on adapte le dossier
        self.adj_name = Generateur.dossier + "/adjectif.txt"
        self.article_name = Generateur.dossier + "/article.txt"
        self.verbe_name = Generateur.dossier + "/verbe.txt"
        self.nom_name = Generateur.dossier + "/nom.txt"
        self.adv_name = Generateur.dossier + "/adverbe.txt"
        self.prep_name = Generateur.dossier + "/preposition.txt"
        
        # Chargement des listes des mots à partir des fichiers
        self.preps = utility.read_file(self.prep_name)
        self.advs = utility.read_file(self.adv_name)
        self.adjs = utility.read_file(self.adj_name)
        self.articles = utility.read_file(self.article_name)
        self.noms = utility.read_file(self.nom_name)
        self.verbes = utility.read_file(self.verbe_name)
        
    
    # Génére un syntagme nominal SNominal = (Dét) Nom (SPrep) (SAdj)
    def gen_syn_nominale(self):
        article = random.choice(self.articles)
        nom = random.choice(self.noms)
        prep = ""
        adj = ""
        
        # Structure optionnel ajoutée au hasard
        if random.random() <0.3:
            prep = self.gen_syn_prep()
        if random.random() <0.3:
            adj = self.gen_syn_adj()
        
        return utility.concat(article, nom, prep, adj)

    # Génère un syntagme verbal SVerb = Verbe (SNominal) 
    def gen_syn_verbale(self):
        verbe = self.gen_verbe_conjugue()
        complement = self.gen_syn_nominale()
        
        return utility.concat(verbe, complement)
    
    # Génère un syntagme adverbial SAdv = Adverbe (SAdv)
    def gen_syn_adv(self):
        adv = random.choice(self.advs)
        synAdv = ""

        if random.random() < 0.2:
            synAdv = self.gen_syn_adv()
        
        return utility.concat(synAdv, adv)

    # Génère un syntagme prépositionnel SPrep = Prep SNominal
    def gen_syn_prep(self):
        prep = random.choice(self.preps)
        syn_nominal = self.gen_syn_nominale()
        
        return utility.concat(prep, syn_nominal)
    
    # Génère un syntagme adjectival SAdj = (SAdv) Adjectif
    def gen_syn_adj(self):
        synAdv = ""
        adj = random.choice(self.adjs)
        
        if random.random() < 0.2:
            synAdv = self.gen_syn_adv()
        
        return utility.concat(synAdv, adj)
    
    # Première amélioration concernant la conjugaison
    def gen_verbe_conjugue(self):
        verbe = random.choice(self.verbes)
        if verbe[-2:] == "er":
            return verbe[:-2] + "e"
        if verbe[-2:] == "ir":
            return verbe[:-2] + "it"
        return self.gen_verbe_conjugue() 
        
    # Génère une phrase Phrase = SNom SVerb
    def gen_phrase(self):
        nominal = self.gen_syn_nominale()
        verbal = self.gen_syn_verbale()
        return utility.concat(nominal, verbal+".") # On ajoute un point pour finir la phrase
    
    # Génère un paragraphe (ie. un ensemble de phrases). On fournit le nombre de phrases que l'on veut
    def gen_paragraphe(self, n):
        phrases = [self.gen_phrase() for i in range(n)]
        return "\n".join(phrases)
    
    # Ecrit sur le fichier de sauvegarde une phrase 
    def output_phrase(self):
        self.output_file.write(self.gen_phrase())
        
    # Ecrit sur le fichier de sauvegarde un paragraphe.
    def output_paragraphe(self,n, commentaire=""):
        texte = "#" + commentaire + "\n\n" + self.gen_paragraphe(n)
        self.output_file.write(texte)
            # On pourra éventuellement un commentaire (date de création, choix des listes ...)
    
    
        
    
    
