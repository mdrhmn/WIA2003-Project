def return_indices_of_a(a, b):
    b_set = set(b)
    return [i for i, v in enumerate(a) if v in b_set]

# Original
# l1 = [1, -2, 3, 4, 6]          
l1 = [3.029, 2.778, -1.449, -1.453, -1.680, -1.768, -1.927]   
# Compared           
l2 = [3.029, 2.778, -1.680, -1.453, -1.449, -1.768, -1.927]    

# l3 = [l2_element for l1_element, l2_element in zip(l1, l2) if l1_element != l2_element]

# for index, (first, second) in enumerate(zip(l1, l2)):
#     if first != second:
#         print(index, second)

# Mismatch sentiment values
incorrect = []   
# Original indexes
original = []
# Sentiment measure
sent_measure = []

for x,y in enumerate(l2):
    if l1[x] != y:

        # If sentiment value < 0 and l2.index < l1.index (to the left)
        if y < 0 and l2.index(y) < l1.index(y):
            sent_measure.append(-2 * (abs(l2.index(y) - l1.index(y))))
        elif y < 0 and l2.index(y) > l1.index(y):
            sent_measure.append(-1 * (abs(l2.index(y) - l1.index(y))))
        elif y > 0 and l2.index(y) < l1.index(y):
            sent_measure.append(-1 * (abs(l2.index(y) - l1.index(y))))
        elif y < 0 and l2.index(y) > l1.index(y):
            sent_measure.append(-2 * (abs(l2.index(y) - l1.index(y))))

        incorrect.append([y,l2.index(y)])  
        original.append(l1.index(y))
  
print(incorrect)
print(original)
print(sent_measure)