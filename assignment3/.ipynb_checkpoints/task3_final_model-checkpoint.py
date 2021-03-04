import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
import torch

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *(torch.tanh(nn.functional.softplus(x)))


class ConvModel1(nn.Module):

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
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            
            #layer 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_filters),
            Mish(),
            nn.Conv2d(in_channels=num_filters,out_channels=64, kernel_size=3,stride=1,padding=1),  
            Mish(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #layer 2
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3,stride=1,padding=1),  
            nn.BatchNorm2d(128),
            Mish(),
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1),  
            Mish(),
            nn.MaxPool2d(kernel_size=2,stride=2),


            #layer 3
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3,stride=1,padding=1),  
            nn.BatchNorm2d(256),
            Mish(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3,stride=1,padding=1),  
            Mish(),
            nn.MaxPool2d(kernel_size=2,stride=2),

        )

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 256*4*4 #convoluted to 128 channels, maxpooled to 4x4 imgs
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features,1024),
            Mish(),
            nn.Linear(1024,1024),
            Mish(),
            nn.Linear(1024, num_classes),
        )


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        layer1 = self.feature_extractor(x)
        features = layer1.view(-1, self.num_output_features)
        out = self.classifier(features)
        out 
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
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

    print(trainer.train_history["accuracy"].popitem(last=True), " train acc" )
    print(trainer.train_history["loss"].popitem(last=True), " train loss" )
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_final_.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
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
    create_plots(trainer, "task2")