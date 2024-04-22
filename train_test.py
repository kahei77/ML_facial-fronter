import collections
import os
import random
from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import FacialExpressionModel

from tqdm.auto import tqdm

import matplotlib.pyplot as plt


TRAIN_DATA_SIZE = 100  # number of images to use in each class of training data
TEST_DATA_SIZE = 30  # number of images to use in each class of testing data
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001


# Training step:
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str
) -> (float, float):
    # Put the model in training mode:
    model.train()

    # Set up training loss and training accuracy values:
    train_loss, train_acc = 0, 0

    # Loop through `dataloader`:
    for batch, (X, y) in enumerate(dataloader):
        # Send data to the target device:
        X, y = X.to(device), y.to(device)

        # Forward pass:
        y_pred = model(X)

        # Calculate and accumulate the loss:
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Optimizer zero grad:
        optimizer.zero_grad()

        # Loss backward:
        loss.backward()

        # Optimizer step:
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches:
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust the metrics to get the average loss and accuracy per batch:
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


# Testing step:
def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: str
) -> (float, float):
    # Put the model in testing mode:
    model.eval()

    # Set up testing loss and testing accuracy values:
    test_loss, test_acc = 0, 0

    with torch.inference_mode():  # Turn on inference mode

        # Loop through `dataloader`:
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the target device:
            X, y = X.to(device), y.to(device)

            # Forward pass:
            test_pred_logits = model(X)

            # Calculate and accumulate the loss:
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy metric across all batches:
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Adjust the metrics to get the average loss and accuracy per batch:
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# Training and testing the model:
def train_test_model(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: str
) -> dict[str, list]:
    # Create a dict for storing the training and testing results:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Do the training and testing steps:
    for epoch in tqdm(range(epochs)):
        # Training:
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        # Testing:
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

        # Update `results`:
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def get_current_time_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S")


def plot_loss_accuracy(results):
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    train_acc = results["train_acc"]
    test_acc = results["test_acc"]
    epochs = range(len(train_loss))

    plt.figure(figsize=(17, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train accuracy")
    plt.plot(epochs, test_acc, label="test accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    current_time_str = get_current_time_str()
    loss_accuracy_plots_path = Path("loss_accuracy_plots")
    if not loss_accuracy_plots_path.exists():
        loss_accuracy_plots_path.mkdir()
    plt.savefig(loss_accuracy_plots_path / f"plot_{current_time_str}.png")


# Make predictions on `image` with loaded state_dict:
def make_predictions(image) -> str:
    model = FacialExpressionModel(
        input_shape=3,
        hidden_units=10,
        output_shape=len(class_names)
    ).to(device)

    models_path = Path("models")
    model_file_name = "model_2024-04-21-12.34.21.pt"  # Change this to the name of the model file you want to use
    model_save_path = models_path / model_file_name
    model.load_state_dict(torch.load(model_save_path))

    model.eval()

    image = data_transform(image)
    with torch.inference_mode():
        image = image.unsqueeze(dim=0)
        image_pred = model(image.to(device))

    image_pred_probs = torch.softmax(image_pred, dim=1)
    image_pred_label = torch.argmax(image_pred_probs, dim=1).cpu()  # Put it onto the cpu to prevent potential errors
    result = class_names[image_pred_label]

    return result


device = "mps" if torch.backends.mps.is_available() else "cpu"

data_path = Path("data/")
train_path = data_path / "train"
test_path = data_path / "test"
image_paths_list = list(data_path.glob("*/*/*.jpg"))

data_transform = transforms.Compose([
    transforms.Resize(size=(48, 48)),
    transforms.ToTensor()
])

train_data = ImageFolder(
    root=train_path,
    transform=data_transform,
    target_transform=None
)

test_data = ImageFolder(
    root=test_path,
    transform=data_transform
)

# Get the class indices of all training data:
train_class_indices = collections.defaultdict(list)
for batch, (X, y) in enumerate(train_data):
    train_class_indices[y].append(batch)

# Get the same number of images in each class of training data:
train_balanced_indices = []
for indices in train_class_indices.values():
    # Randomly pick `TRAIN_DATA_SIZE` sample indices from the current class of images:
    sample_indices = random.sample(indices, TRAIN_DATA_SIZE)
    # Append `sample_indices` to `balanced_indices`:
    train_balanced_indices.extend(sample_indices)

# Create balanced training datasets:
balanced_train_data = Subset(train_data, train_balanced_indices)

# Get the class indices of all testing data:
test_class_indices = collections.defaultdict(list)
for batch, (X, y) in enumerate(test_data):
    test_class_indices[y].append(batch)

# Get the same number of images in each class of testing data:
test_balanced_indices = []
for indices in test_class_indices.values():
    # Randomly pick `TEST_DATA_SIZE` sample indices from the current class of images:
    sample_indices = random.sample(indices, TEST_DATA_SIZE)
    # Append `sample_indices` to `balanced_indices`:
    test_balanced_indices.extend(sample_indices)

# Create balanced training datasets:
balanced_test_data = Subset(test_data, test_balanced_indices)

# Get a list of class names:
class_names = train_data.classes

# Turn the training and testing datasets into `DataLoader`'s:
train_dataloader = DataLoader(
    dataset=balanced_train_data,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    shuffle=True
)
test_dataloader = DataLoader(
    dataset=balanced_test_data,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    shuffle=False
)

model = FacialExpressionModel(
    input_shape=3,
    hidden_units=10,
    output_shape=len(class_names)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

if __name__ == '__main__':

    # Train the model and get the results:
    model_results = train_test_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device
    )

    # Plot the loss and accuracy curves:
    plot_loss_accuracy(model_results)

    # Save the model's `state_dict`:
    current_time_str = get_current_time_str()
    model_file_name = f"model_{current_time_str}.pt"
    models_path = Path("models")
    if not models_path.exists():
        models_path.mkdir()
    model_save_path = models_path / model_file_name
    torch.save(model.state_dict(), model_save_path)
