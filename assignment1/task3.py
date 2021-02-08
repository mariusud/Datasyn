import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)
plt.style.use('seaborn-white')

def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    
    y = model.forward(X)
    y = np.argmax(y,axis=1)
    y_true = np.argmax(targets,axis=1)
    correct = np.isclose(y_true,y,atol=0)
    return np.sum(correct) / y.shape[0]


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        y = self.model.forward(X_batch)
        self.model.backward(X_batch,y,Y_batch) # compute the gradient 
        delta_W = self.model.grad
        self.model.w = self.model.w - self.learning_rate*delta_W #perform gradient descent step
        loss = cross_entropy_loss(Y_batch,y)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True
    early_stop = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset, early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)

    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)

    fig, axs = plt.subplots(2,10, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.suptitle('Digit weights. Top lambda=0, bottom lambda=1', fontsize=20)
    axs = axs.ravel()
    for i in range(10):
        im = model.w[:-1,i].reshape(28,28)
        axs[i].imshow(im,cmap='gray')
    for j in range(10):
        im2 = model1.w[:-1,j].reshape(28,28)
        axs[j+10].imshow(im2,cmap='gray')
    plt.show()


    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]



    model2 = SoftmaxModel(l2_reg_lambda=0.1)
    trainer = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg02, val_history_reg02 = trainer.train(num_epochs)

    model3 = SoftmaxModel(l2_reg_lambda=0.01)
    trainer = SoftmaxTrainer(
        model3, learning_rate, batch_size, shuffle_dataset,early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg03, val_history_reg03 = trainer.train(num_epochs)


    model4 = SoftmaxModel(l2_reg_lambda=0.001)
    trainer = SoftmaxTrainer(
        model4, learning_rate, batch_size, shuffle_dataset,early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg04, val_history_reg04 = trainer.train(num_epochs)


    # Plot accuracy
    plt.ylim([0.72, .93])
    utils.plot_loss(val_history_reg01["accuracy"], "Validation Accuracy for lambda=1")
    utils.plot_loss(val_history_reg02["accuracy"], "Validation Accuracy for lambda=0.1")
    utils.plot_loss(val_history_reg03["accuracy"], "Validation Accuracy for lambda=0.01")
    utils.plot_loss(val_history_reg04["accuracy"], "Validation Accuracy for lambda=0.001")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight

    plt.savefig("task4d_l2_reg_norms.png")

    norm1 = (np.sum(model1.w*model1.w))
    norm2 = (np.sum(model2.w*model2.w))
    norm3 = (np.sum(model3.w*model3.w))
    norm4 = (np.sum(model4.w*model4.w))


    lambdas = [1, .1, .01, .001]
    norms = [norm1, norm2, norm3, norm4]
    plt.xlim([0.001, 1.1])

    plt.plot(lambdas, norms)
    plt.xlabel("Corresponding lambda")
    plt.ylabel("L2 norm")
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()