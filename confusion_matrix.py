conf_matrix = [ [0]*3 for _ in range(3) ]

for line in file("predictions.txt"):
    predicted, actual = map(int, line.strip().split(','))
    conf_matrix[(actual-1)][(predicted-1)] += 1

for x in range(3):
    for y in range(3):
        print conf_matrix[x][y], "\t",
    print "