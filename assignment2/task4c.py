import numpy as np
import utils
import matplotlib.pyplot as plt
from task2 import SoftmaxTrainer
import numpy as np

from task2a import one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test
plt.style.use('seaborn-white')


if __name__ == "__main__":
    '''
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Modify your network here
    neurons_per_layer = [60, 60, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    #gradient_approximation_test(model, X_train, Y_train)
    '''

    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True
    early_stopping = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    neurons_per_layer = [60, 60, 10]
    neurons_per_layer_small = [64,10]
    neurons_per_layer_huge = [64,64,64,64,64,64,64,64,64,64,10]
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
    model_one_layer = SoftmaxModel(
        neurons_per_layer_small,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_one_layer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_one_layer, learning_rate, batch_size, shuffle_data, early_stopping,
        X_train, Y_train, X_val, Y_val,
    )

    model_ten_layer = SoftmaxModel(
        neurons_per_layer_huge,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_ten_layer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_ten_layer, learning_rate, batch_size, shuffle_data, early_stopping,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_ten, val_history_ten = trainer_ten_layer.train(num_epochs)
    train_history_one, val_history_one = trainer_one_layer.train(num_epochs)
    train_history, val_history = trainer.train(num_epochs)

    plt.figure(figsize=(20,12))
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Training Loss model with 2 layers", npoints_to_average=10)
    utils.plot_loss(
        train_history_one["loss"], "Training Loss model with 1 layers", npoints_to_average=10)
    utils.plot_loss(
        train_history_ten["loss"], "Training Loss model with 10 layers", npoints_to_average=10)

    
    utils.plot_loss(val_history["loss"],
                "Validation loss model with 2 layers", npoints_to_average=10)
    utils.plot_loss(
        val_history_one["loss"], "Validation loss model with 1 layers", npoints_to_average=10)
    utils.plot_loss(
        val_history_ten["loss"], "Validation loss model with 10 layers", npoints_to_average=10)

    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1.01])
    utils.plot_loss(val_history["accuracy"], "Validation accuracy model with 2 layers")
    utils.plot_loss(
        val_history_one["accuracy"], "Validation accuracy model with 1 layer")
    utils.plot_loss(
        val_history_ten["accuracy"], "Validation accuracy model with 10 layers")

    utils.plot_loss(train_history["accuracy"], "Training accuracy model with 2 layers")
    utils.plot_loss(
        train_history_one["accuracy"], "Training accuracy model with 1 layer")
    utils.plot_loss(
        train_history_ten["accuracy"], "Training accuracy model with 10 layers")

    
    plt.ylabel("Accuracy")
    plt.xlabel("Step")

    plt.legend()
    plt.savefig("task4c.png")
    plt.show()