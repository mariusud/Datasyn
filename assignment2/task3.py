import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import numpy as np
plt.style.use('seaborn-white')

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True
    early_stop = True
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train, mean, std)

    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data, early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    ## RUN WITH MOMENTUM
    use_momentum = True
    model_momentum = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_momentum, learning_rate, batch_size, shuffle_data, early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_momentum, val_history_momentum = trainer_momentum.train(
        num_epochs)


    ## RUN WITH IMPROVED SIGMOID
    use_momentum = False
    use_improved_sigmoid = True
    model_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_sigmoid, learning_rate, batch_size, shuffle_data, early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_sigmoid, val_history_sigmoid = trainer_sigmoid.train(
        num_epochs)

    ## RUN WITH IMPROVED WEIGHT
    use_improved_sigmoid = True
    use_improved_weight_init = True

    model_weight = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_weight = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_weight, learning_rate, batch_size, shuffle_data, early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_weight, val_history_weight = trainer_weight.train(
        num_epochs)


    ## FINAL RUN
    use_momentum = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    model_final = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_final = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_final, learning_rate, batch_size, shuffle_data, early_stop,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_final, val_history_final = trainer_final.train(
        num_epochs)

    # PLOT
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"], "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(train_history_momentum["loss"], "Added Momentum model", npoints_to_average=10)
    utils.plot_loss(train_history_sigmoid["loss"], "Improved sigmoid function", npoints_to_average=10)
    utils.plot_loss(train_history_weight["loss"], "Imrpoved weight init", npoints_to_average=10)
    utils.plot_loss(train_history_final["loss"], "All improvements included", npoints_to_average=10)

    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])


    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(val_history_momentum["accuracy"], "Added Momentum")
    utils.plot_loss(val_history_sigmoid["accuracy"], "Improved sigmoid function")
    utils.plot_loss(val_history_weight["accuracy"], "Improved weight init")
    utils.plot_loss(val_history_final["accuracy"], "All improvements included")
    
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task3.png")

    plt.show()
