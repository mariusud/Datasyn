import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import numpy as np

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer_small = [32, 10]
    neurons_per_layer_large = [128, 10]

    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    early_stopping = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer_small,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data, early_stopping,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created in assignment text - Comparing with and without shuffling.
    # YOU CAN DELETE EVERYTHING BELOW!
    shuffle_data = False
    model_large = SoftmaxModel(
        neurons_per_layer_large,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_large, learning_rate, batch_size, shuffle_data, early_stopping,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)
    shuffle_data = True
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Model with 32 neurons", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Model with 128 neurons", npoints_to_average=10)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()

    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])




    utils.plot_loss(val_history["accuracy"], "Model with 32 neurons")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Model with 128 neurons")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4ab.png")

    plt.show()