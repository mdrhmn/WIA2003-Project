# theList = [(1, 'a'), (1, 'b'), (5, 'c')]

# b = sorted(sorted(theList, key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

# print(b)

a =[('Al',2),('Bill',1),('Carol',2),('Abel',3),('Zeke',2),('Chris',1)]
b = sorted(sorted(a, key =lambda x : x[0]), key =lambda x : x[1], reverse =True)
print(b)
