import torch
from torch import nn, optim
from model import NetWork
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Define the image transformations: convert to grayscale and tensor
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
    )

    # Load the training dataset and apply transformations
    train_dataset = datasets.ImageFolder(root="mnist_train", transform=transform)
    # Load the test dataset and apply transformations
    test_dataset = datasets.ImageFolder(root="mnist_test", transform=transform)
    print("Train dataset size: ", len(train_dataset))
    print("Test dataset size: ", len(test_dataset))

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Train loader size: ", len(train_loader))

    # Iterate over the first few bathces of the training dataloader
    for batch_index, (data, label) in enumerate(train_loader):
        # Uncomment the following lines to break after 3 batches
        # if batch_idx == 3:
        #     break
        print("Batch index: ", batch_index)
        print("Data shape: ", data.shape)
        print("Label shape: ", label.shape)
        print(label)

    # Initialize the neural network model
    model = NetWork()
    # Initialize the Adam optimizer with the model's parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define the loss function as cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Train the model for 10 epochs
    for epoch in range(10):
        # Iterate over the batches in the training DataLoader
        for batch_index, (data, lavel) in enumerate(train_loader):
            # Forward pass: compute the model output
            output = model(data)
            # Compute the loss
            loss = criterion(output, label)
            # Backward pass: compute the gradients
            loss.backward()
            # Update the model parameters
            optimizer.step()
            # Zero the gradients for the next iteration
            optimizer.zero_grad()
            # Print the loss every 100 batches
            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/10"
                    f"Batch {batch_index}/{len(train_loader)}"
                    f"Loss {loss.item():.4f}"
                )
    # Save the trained model's state dictionary to a file
    torch.save(model.state_dict(), "model.pth")
