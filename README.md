# TIPE-ML

Ce TIPE a pour but d'étudier la génération automatique de texte par la création de texte aléatoire cohérent. A partir d'un corpus initial de texte, on entraine un alogrithme qui par la suite pourra des textes intelligibles. Pour cela, plusieurs pistes sont étudiés (ayant des complexités d'implémentations et des performances différentes). Les solutions retenues sont les suivantes :
- La génération suivant un schéma de Markov 
- La génération suivant une structure de langage issue de la classification des langages de Chomsky
- La génération par réseau de neuronne 

Motivation:
La publication d'études scientifiques dans des revues est devenue la norme dans le domaine scientifique. Le nombre de citation et d'articles publiés sont ainsi devenus des mesures de la renommée scientifique. Toutefois, ces études peuvent être falsifiées à l'aide de l'outils numériques. Nous allons ainsi voir à quel point il est possible de générer du texte. La prévention et la détection de tels artifices seront nécessaires pour affirmer la légitimité des auteurs dont elle porte le nom.



Traitement du Langage Naturel (TLN):
ou NLP (Natural Language Processing) en anglais

Algorithmes connus en TLN

- Schéma de Markov : Solution naïve
On construit le texte de façon itérative mot à mot. A chaque mot (ou vecteur de mot), on associe une certaine probabilité d'avoir un prochain mot. L'entrainement de l'algorithme se fait à partir d'un corpus de texte. On s'appuie sur un schéma de Markov d'ordre 1 (pas sûr que ce terme existe vraiment)
On extrait le vocabulaire. Pour chaque mot présent dans le texte, on regarde l'ensemble des mots suivants que l'on dénombre avec multiplicité. On associe ensuite une probabilité suivant la multiplicité.
On peut ainsi générer un texte de longueur arbitraire en fournissant un premier mot (présent dans le vocabulaire).

Possibilités:
- S'arrêter lorsqu'on rencontre un point
- On pourra changer d'ordre en se ramenant à des vecteurs de mots (la complexité explose, en O(len(vocabulaire) ** n) où n désigne est l'ordre choisie (on s'intéresse à la suite de n mots)

Mise en place :
- On utilise un dictionnaire pour faire le dénombrement en parcourant le texte mot à mot
- On choisit au hasard le prochain mot et on continue

Avantage:
- Simplicité de mise en place (entrainement)
- On a l'espoir que l'on puisse retrouver du sens parce que l'enchainement des mots provient de vrais phrases

Inconvénient :
- Beh ça marche pas super bien 

Ce que je prévois pour cette solution :
- Markov donnera surement de meilleure texte si on utilise un corpus qui se base sur le même thème. On espère alors que les phrases seront tous sur ce thème et puissent s'enchainer assez bien.
-> A tester (longueur des textes utilisés, corpus sur le même thème ou pas 

- Structure de Chomsky du langage :
Une structure grammaticale (simplifiée) du langage natural selon le travail d

Lien :
SCIgen : créant de fausses études scientifiques
https://pdos.csail.mit.edu/archive/scigen/
Le code source :
https://github.com/strib/scigen