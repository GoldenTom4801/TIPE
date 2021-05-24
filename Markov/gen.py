import markov
import markovOpti
import time

with markovOpti.GenMarkov("Markov/data/fables_fontaine.txt", "Markov/output/test_fables_fontaine3.txt") as gen:
    #print(gen.genParagraphe("les"))

    print(gen.genParagraphe("les", 100))