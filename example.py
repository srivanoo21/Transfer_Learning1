def addition(a, b):
    result = a + b
    return result

def multiplication(a, b):
    result = a * b
    return result

def division(a, b):
    result = a / b
    return result

# (2*3) + 4 = 10
#print(addition(multiplication(2, 3), 4))

# (1*2) + 3 = 5
#print(addition(multiplication(1, 2), 3))

print("((5+2)*10)/2 is", division(multiplication(addition(5, 2), 10), 2))