import numpy as np
import utils
import typing
from tqdm import tqdm
np.random.seed(1)

def softmax(z):
    """The Softmax function."""
    return np.exp(z) / np.sum(np.exp(z), axis=1,keepdims=True)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def improved_sigmoid(z):
    """The sigmoid function."""
    return 1.7159*np.tanh(2.0*z/3.0)

def improved_sigmoid_derivative(z):
    """The sigmoid function."""
    return 2.0*1.7159 / (3.0*(np.cosh(((2.0/3.0)*z))**2))

def sigmoid_derivative(z):
    """The sigmoid derivative function."""
    return sigmoid(z)*(1-sigmoid(z))

def pre_process_images(X: np.ndarray, mean: int, std: int):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    '''
    means =[np.mean(image) for image in X]
    mean = np.sum(means) / X.shape[0]

    stds = [np.std(image) for image in X]
    std = np.sum(stds) / X.shape[0]
    '''
    X = (X - mean )/std
    bias = np.ones((X.shape[0],1))
    X = np.append(X, bias, axis=1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return np.sum(-np.sum(targets*np.log(outputs),axis=1)) / targets.shape[0]


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        if use_improved_weight_init:
            self.ws = []
            prev = self.I
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                print("Initializing improved weight to shape:", w_shape)
                w = np.random.normal(0, 1/np.sqrt(prev),w_shape)
                self.ws.append(w)
                prev = size
        else:
            self.ws = []
            prev = self.I
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                print("Initializing weight to shape:", w_shape)
                w = np.random.uniform(-1,1,w_shape)
                self.ws.append(w)
                prev = size


        self.grads = [None for i in range(len(self.ws))]
        self.d = [[None] for i in range(len(self.neurons_per_layer))]
        self.zs = [None] * (len(self.neurons_per_layer)+1)
        self.activations = list()

    def activation_function(self, z):
        if self.use_improved_sigmoid:
            return improved_sigmoid(z)
        else:
            return sigmoid(z)

    def activation_function_derivative(self,z):
        if self.use_improved_sigmoid:
            return improved_sigmoid_derivative(z)
        else:
            return sigmoid_derivative(z)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For peforming the backward pass, you can save intermediate activations in varialbes in the forward pass.
        # such as self.hidden_layer_ouput = ...
        self.activations = [X] 
        for i in range(len(self.ws)-1):
            #predict  layer
            z = self.activations[i].dot(self.ws[i]) # z_j = weights @ x
            #activate  layer
            activation = self.activation_function(z) # a_j = f(z_j) 
            self.zs.append(z)
            self.activations.append(activation)
        return softmax(activation.dot(self.ws[-1])) #final step a_k

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        self.grads = [np.zeros(w.shape) for w in self.ws]

        # Initialize error of output layer
        output_error = - (targets-outputs) / targets.shape[0] #error from final layer
        self.grads[-1] = np.dot(self.activations[-1].T, output_error) # calculate gradient for last layer

        error_vector = output_error # first error is simply the output error of the NN
        # start to backpropagate the error through the layers
        for i in range(1,len(self.neurons_per_layer)):
            # calculate the error vector  delta_l for layer
            error_vector = np.dot(error_vector,self.ws[-i].T) * self.activation_function_derivative(self.zs[-i]) # BP2
            # calculate gradient
            self.grads[-i-1] =  np.dot(self.activations[-i-1].T, error_vector) # BP4

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    arr = np.zeros((Y.shape[0],num_classes),dtype=int)
    for example in range(Y.shape[0]):
        label = Y[example]
        arr[example, label]=1
    return arr


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in tqdm(range(w.shape[0])):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = X_train[:1]
    Y_train = Y_train[:1]

    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train,mean,std)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)