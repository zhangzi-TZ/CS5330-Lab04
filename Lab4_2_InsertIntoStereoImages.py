import cv2
import numpy as np
from PIL import Image
from Lab4_1_SegmentImage import segment_person, load_model

# Load the stereo image and split it into left and right images
def load_stereo_image(image_input):
    # Check the input type
    if isinstance(image_input, str):  # If it's a file path, load the image
        stereo_image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):  # If it's already a PIL.Image object, use it directly
        stereo_image = image_input
    else:
        raise ValueError("Invalid image input type. Should be a file path or a PIL.Image object.")

    width, height = stereo_image.size

    # Split the image into left and right images
    left_image = stereo_image.crop((0, 0, width // 2, height))
    right_image = stereo_image.crop((width // 2, 0, width, height))

    return left_image, right_image

# Insert the person image into the left and right images, adding a position parameter
def insert_person(left_image, right_image, person_image, depth="close", position=(50, 50)):
    # Adjust horizontal offset based on depth
    if depth == "close":
        offset = 20  # Close distance: larger offset
    elif depth == "medium":
        offset = 10  # Medium distance: medium offset
    elif depth == "far":
        offset = 5   # Far distance: smaller offset
    else:
        offset = 20  # Default to close distance

    # Convert PIL images to NumPy arrays, keeping RGB format
    left_image_cv = np.array(left_image)
    right_image_cv = np.array(right_image)
    person_image_cv = np.array(person_image)

    # Extract the RGB and alpha channels of the person image
    rgb_person = person_image_cv[:, :, :3]  # Extract RGB channels
    alpha_mask = person_image_cv[:, :, 3]   # Extract alpha channel as mask

    # Calculate insertion position
    max_x = left_image_cv.shape[1] - person_image_cv.shape[1]
    max_y = left_image_cv.shape[0] - person_image_cv.shape[0]
    pos_x = int(max_x * (position[0] / 100))
    pos_y = int(max_y * (position[1] / 100))

    # Ensure the insertion position is within the image bounds
    pos_x = max(0, min(pos_x, max_x))
    pos_y = max(0, min(pos_y, max_y))

    # Compute the actual size of the insertion area (may adjust due to boundaries)
    person_h, person_w = person_image_cv.shape[0], person_image_cv.shape[1]

    # Left image insertion area
    left_insert_x1 = pos_x
    left_insert_x2 = pos_x + person_w
    left_insert_y1 = pos_y
    left_insert_y2 = pos_y + person_h

    # If insertion area exceeds image boundaries, adjust insertion area and person image crop
    if left_insert_x2 > left_image_cv.shape[1]:
        excess = left_insert_x2 - left_image_cv.shape[1]
        left_insert_x2 -= excess
        person_w -= excess
        rgb_person = rgb_person[:, :person_w, :]
        alpha_mask = alpha_mask[:, :person_w]

    if left_insert_y2 > left_image_cv.shape[0]:
        excess = left_insert_y2 - left_image_cv.shape[0]
        left_insert_y2 -= excess
        person_h -= excess
        rgb_person = rgb_person[:person_h, :, :]
        alpha_mask = alpha_mask[:person_h, :]

    # Insert into the left image
    for c in range(3):  # Process each RGB channel separately
        left_image_cv[left_insert_y1:left_insert_y2, left_insert_x1:left_insert_x2, c] = \
            left_image_cv[left_insert_y1:left_insert_y2, left_insert_x1:left_insert_x2, c] * (1 - alpha_mask / 255) + \
            rgb_person[:, :, c] * (alpha_mask / 255)

    # Right image insertion area
    pos_x_right = pos_x + offset
    max_x_right = right_image_cv.shape[1] - person_w
    pos_x_right = max(0, min(pos_x_right, max_x_right))

    right_insert_x1 = pos_x_right
    right_insert_x2 = pos_x_right + person_w
    right_insert_y1 = pos_y
    right_insert_y2 = pos_y + person_h

    if right_insert_x2 > right_image_cv.shape[1]:
        excess = right_insert_x2 - right_image_cv.shape[1]
        right_insert_x2 -= excess
        person_w -= excess
        rgb_person = rgb_person[:, :person_w, :]
        alpha_mask = alpha_mask[:, :person_w]

    if right_insert_y2 > right_image_cv.shape[0]:
        excess = right_insert_y2 - right_image_cv.shape[0]
        right_insert_y2 -= excess
        person_h -= excess
        rgb_person = rgb_person[:person_h, :, :]
        alpha_mask = alpha_mask[:person_h, :]

    # Insert into the right image
    for c in range(3):  # Process each RGB channel separately
        right_image_cv[right_insert_y1:right_insert_y2, right_insert_x1:right_insert_x2, c] = \
            right_image_cv[right_insert_y1:right_insert_y2, right_insert_x1:right_insert_x2, c] * (1 - alpha_mask / 255) + \
            rgb_person[:, :, c] * (alpha_mask / 255)

    # Convert the combined results back to PIL images
    left_image_with_person = Image.fromarray(left_image_cv)
    right_image_with_person = Image.fromarray(right_image_cv)

    return left_image_with_person, right_image_with_person

# Verify the insertion result
if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Load the input image as a PIL.Image object
    input_image = Image.open("input_image.jpg")

    # Perform segmentation
    person_image = segment_person(input_image, model)

    # Load and split the stereo image
    left_image, right_image = load_stereo_image("sbs_neu.jpg")

    # Insert the person image and save the result, example position is (50, 50)
    left_with_person, right_with_person = insert_person(left_image, right_image, person_image, position=(50, 50))

    # Save the left and right images
    left_with_person.save("left_with_person.png")
    right_with_person.save("right_with_person.png")
    print("Left and right images saved as left_with_person.png and right_with_person.png")