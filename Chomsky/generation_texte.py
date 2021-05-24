import chomsky
import time

with chomsky.Generateur("Chomsky/test_output/test.txt") as gen:
    deb = time.time()
    gen.output_paragraphe(100, "23/05")
    print(time.time() - deb)
    #print(gen.gen_phrase())
    