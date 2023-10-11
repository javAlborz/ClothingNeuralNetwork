# Import necessary libraries and modules
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import helper
import fc_model
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import argparse


def process_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert the colors
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize the image to 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension



# Define the model, criterion and optimizer
model = fc_model.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Uncomment below lines to retrain the model
# fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=20)
# torch.save(model.state_dict(), 'checkpoint_new.pth')

# Running inference on provided image
state_dict = torch.load('checkpoint.pth')
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()



def run_inference_and_plot(image_path):
    """
    Takes akes an image path, processes the image, runs inference using a pre-trained model,
    plots the original, preprocessed images and the results, and returns the result.
    """
    # Load and preprocess the image
    original_image = Image.open(image_path)
    preprocessed_img = process_image(image_path).squeeze()
    
    # Run inference
    img = preprocessed_img.view(1, 784)
    with torch.no_grad():
        output = model.forward(img)
    ps = torch.exp(output)

    # Plotting
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.5])
    fig = plt.figure(figsize=(10, 10))

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2:])
    
    ax0.imshow(original_image)
    ax0.set_title("Original Image")
    ax0.axis('off')

    border_thickness = 5  
    border_color = 'black' 
    img_width, img_height = original_image.size  
    rect = patches.Rectangle((0-border_thickness, 0-border_thickness), img_width+2*border_thickness, img_height+2*border_thickness, linewidth=border_thickness, edgecolor=border_color, facecolor='none')
    ax0.add_patch(rect)

    ax1.imshow(preprocessed_img.numpy())
    ax1.set_title("Preprocessed Image")
    ax1.axis('off')

    helper.view_classify(img.view(1, 28, 28), ps, version='Fashion', ax=ax2)
    plt.tight_layout()
    plt.show()

    return ps  # or another appropriate return value




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image for inference.')
    args = parser.parse_args()
    image_path = args.image_path

    # Then you can call your functions within this block with the provided image_path
    processed_image = process_image(image_path)
    run_inference_and_plot(image_path)





# # --Running inference on random images from test set--

# # Define transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # Download and load the training data
# trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# # Download and load the test data
# testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)



# state_dict = torch.load('checkpoint.pth')
# print(state_dict.keys())
# model.load_state_dict(state_dict)

# model.eval()

# dataiter = iter(testloader)
# images, labels = next(dataiter)
# img = images[0]

# # Convert 2D image to 1D vector
# img_1d = img.view(1, 784)

# # Calculate the class probabilities (softmax) for img
# with torch.no_grad():
#     output = model.forward(img_1d)

# ps = torch.exp(output)

# # Create subplots
# fig = plt.figure(figsize=(10, 10))
# gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.5])

# # Original Image (Before Preprocessing)
# ax0 = plt.subplot(gs[0])
# ax0.imshow(img.numpy().squeeze(), cmap='gray')  # Assuming the image is grayscale
# ax0.set_title("Original Image")
# ax0.axis('off')

# # Add border around the original image
# rect = patches.Rectangle((-0.5, -0.5), 28, 28, linewidth=2, edgecolor='black', facecolor='none')
# ax0.add_patch(rect)

# # Preprocessed Image
# ax1 = plt.subplot(gs[1])
# # Display the img_1d
# ax1.imshow(img.numpy().squeeze())  # Display the img_1d if you don't reverse the preprocessing
# ax1.set_title("Preprocessed Image")
# ax1.axis('off')

# # Classification Results
# ax2 = plt.subplot(gs[2:])
# helper.view_classify(img.view(1, 28, 28), ps, version='Fashion', ax=ax2)

# plt.tight_layout()
# plt.show()

