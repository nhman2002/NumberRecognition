import torch
from model import NetWork
from torchvision import transforms, datasets

if __name__ == "__main__":
    # Define the image transformations: convert to grayscale and then to tensor
    transforms = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
    )
    # Load the test dataset from the specified directory and apply transformations
    test_datasets = datasets.ImageFolder(root="mnist_test", transform=transforms)
    print("test_datasets size: ", len(test_datasets))

    model = NetWork()
    # Load the model's state dictionary from the saved file
    model.load_state_dict(torch.load("model.pth"))

    right = 0   # Initialize a counter for correctly classified images

    for i, (x, y) in enumerate(test_datasets):
        output = model(x.unsqueeze(0))  # Forward pass: add batch dimension and compute the model output
        predicted = output.argmax(1).item() # Get the index of the highest score as the predicted label
        if predicted == y:
            right += 1  # Increment the counter if the prediction is correct
        else:
            img_path = test_datasets.samples[i][0]  # Get the path of the misclassified image

            print(
                f"wrong case: predict = {predicted} actual = {y}, img_path = {img_path}"
            )

    sample_num = len(test_datasets) # Get the total number of samples in the test dataset
    acc = right * 1.0 / sample_num  # Calculate the accuracy as the ratio of correct predictions

    print(f"test accuracy: %d / %d = %.31f" % (right, sample_num, acc))
