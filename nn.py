import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.bias1 = np.full((4,1), 1)
        self.bias2 = np.full((4,1), 1)
        self.output = np.zeros(y.shape, dtype=float)
    
    def feedforward(self):
        self.layer1 = sigmoid(np.add(np.dot(self.input, self.weights1), self.bias1))
        self.output = sigmoid(np.add(np.dot(self.layer1, self.weights2), self.bias2))
        return self.output

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)*sigmoid_derivative(self.output)))
        d_bias2 =  2*(self.y - self.output)*sigmoid_derivative(self.output)
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1)))
        d_bias1 = np.dot(2*(self.y - self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1)

        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.bias1 = np.add(self.bias1, d_bias1)
        self.bias2 = np.add(self.bias2, d_bias2)

    def train(self, iterations):
        for i in range(iterations):
            self.output = self.feedforward()
            self.backprop()
            
            if i % 100 == 0 or i == iterations-1:
                print("Iteration:", i)
                print("Loss:", np.mean(np.square(y - nn.output)), "\n\n")
        
        print("\n##########################Output##########################\n")
        print("Input:\n", x, "\n")
        print("Actual output:\n", y, "\n")
        print("Final Output:\n", nn.output, "\n")

if __name__ == "__main__":
    x = np.array(([0,0], [0, 1], [1, 0], [1, 1]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

    nn = NeuralNetwork(x, y)
    nn.train(2000)
