def perceptron_AND(x1, x2):
    return int(x1 and x2)

def perceptron_OR(x1, x2):
    return int(x1 or x2)

def perceptron_XOR(x1, x2):
    y1 = perceptron_OR(x1, x2)
    y2 = perceptron_AND(x1, x2)
    return y1 - y2

for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"XOR({x1}, {x2}) = {perceptron_XOR(x1, x2)}")
