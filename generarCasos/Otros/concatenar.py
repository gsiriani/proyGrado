import sys

a_1 = open(sys.argv[1],"r")
a_2 = open(sys.argv[2],"r")
a_3 = open(sys.argv[3],"w")

for l in a_1:
	a_3.write(l)
for l in a_2:
	a_3.write(l)

a_1.close()
a_2.close()
a_3.close()