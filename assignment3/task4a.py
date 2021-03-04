
class ResNet(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fully_connected = torch.nn.Linear(512, 10)
        )


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        out = self.model(x)
        return out
       
       
       


def create_plots(trainer: Trainer, name: str):
    test_loss, test_acc = compute_loss_and_accuracy(trainer.dataloader_test, trainer.model, trainer.loss_criterion)
    print(test_loss, test_acc)
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer.train_history["accuracy"], label="Training Accuracy")
    
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_final_.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4 # 5e-4?
    early_stop_count = 10
    dataloaders = load_cifar10(batch_size)
    model = ConvModel1(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task4")