import markov



def evaluation(path):
    gen = markov.GenMarkov("data/fables_fontaine.txt", "void")
    
    f = open(path, "w")
    
    phraseGeniale = []
    phraseAcc = []
    phraseRef = []
    
    phrase = gen.genPhrase("les")
    print(phrase)
    a = input("Garder (G/O/N), Quitter : Q")
    while a!="Q":
        if a == "O":
            nbSuccMoyen = gen.nbMoyenSuccesseur(phrase)
            recopieOriginal, tripletFautif = gen.mesurePhrase(phrase)
                #Un couple de Booléen et String : True si un ensemble de trois mots est présent dans le texte
                # False sinon
            phraseAcc.append(phrase + "\nUn triplet présent? {0} Fautif : {1} Nombre moyen : {2}".format(recopieOriginal, tripletFautif, nbSuccMoyen))
        elif a == "N":
            nbSuccMoyen = gen.nbMoyenSuccesseur(phrase)
            recopieOriginal, tripletFautif = gen.mesurePhrase(phrase)
                #Un couple de Booléen et String : True si un ensemble de trois mots est présent dans le texte
                # False sinon
            phraseRef.append(phrase + "\nUn triplet présent? {0} Fautif : {1} Nombre moyen : {2}".format(recopieOriginal, tripletFautif, nbSuccMoyen))
        elif a =='G':
            nbSuccMoyen = gen.nbMoyenSuccesseur(phrase)
            recopieOriginal, tripletFautif = gen.mesurePhrase(phrase)
                #Un couple de Booléen et String : True si un ensemble de trois mots est présent dans le texte
                # False sinon
            phraseGeniale.append(phrase + "\nUn triplet présent? {0} Fautif : {1} Nombre moyen : {2}".format(recopieOriginal, tripletFautif, nbSuccMoyen))
        else:
            print("Erreur, phrase abandonnée")
        phrase = gen.genPhrase("les")
        print(phrase)
        a = input("Garder (G/O/N), Quitter : Q")
    
    parag_genial = "Phrases géniales" + "\n".join(phraseGeniale)
    parag_acc = "Phrases considérées comme satisfaisantes (au moins en partie) :\n" + "\n".join(phraseAcc)
    parag_ref =  "Phrases non satisfaisantes :\n" + "\n".join(phraseRef)
    texte =  parag_genial + "\n\n\n\n" + parag_acc + "\n\n\n\n" + parag_ref 
    nbTot = len(phraseAcc) + len(phraseRef) + len(phraseGeniale)
    tex_commenté = texte + "\n\n" + "Nombre de phrases total = {0} \n{1}% génial {2}% accepté {3}% refusé".format(nbTot, len(phraseGeniale)/nbTot*100, len(phraseAcc)/nbTot*100, len(phraseRef)/nbTot*100)
    f.write(tex_commenté)
    
    
    
path = "output/evaluation2/phrases.txt"   
evaluation(path)