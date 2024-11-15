import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

# Load the DeepLabV3 model
def load_model():
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    model.eval()
    return model

# Image segmentation function
def segment_person(input_image, model):
    # Ensure the input image is in RGB format
    input_image = input_image.convert("RGB")

    # Preprocess the image
    preprocess = transforms.ToTensor()
    input_tensor = preprocess(input_image).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()

    # Create an image with an alpha channel
    segmented_image = np.array(input_image)

    # Apply the mask as the alpha channel
    alpha_channel = np.where(mask == 15, 255, 0).astype(np.uint8)  # 15 represents the 'person' class
    segmented_image = np.dstack((segmented_image, alpha_channel))

    return Image.fromarray(segmented_image)

# Verify the segmentation result
if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Load the image as a PIL.Image object
    input_image = Image.open("input_image.jpg")

    # Perform segmentation and save the result
    segmented_image = segment_person(input_image, model)

    # Save the segmentation result locally
    output_path = "segmented_person_output.png"
    segmented_image.save(output_path)
    print(f"Segmented image saved to the current directory as {output_path}")