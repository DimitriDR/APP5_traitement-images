import numpy

# 1
cpt = 0
for i in range(1, 100, 2):
    cpt += 1
    print(str(i) + " " + str(cpt) + "e itération") # plus la derniere itération

cpt = 0
for i in numpy.arange(1, 100, 0.5):
    cpt += 1
    print(str(i) + " " + str(cpt) + "e itération")

# 2
cpt = 0
for i in numpy.arange(5, 100, 1):
    cpt += 1
    print(str(i) + " " + str(cpt) + "e itération")