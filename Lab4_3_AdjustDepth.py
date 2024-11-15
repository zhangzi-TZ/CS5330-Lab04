import numpy as np
from PIL import Image, ImageEnhance
from Lab4_1_SegmentImage import segment_person, load_model
from Lab4_2_InsertIntoStereoImages import load_stereo_image, insert_person

# Define horizontal offsets and scaling factors for each depth level
depth_configs = {
    "close": {"offset": 20, "scale": 1.2},   # Close distance, enlarge the person
    "medium": {"offset": 10, "scale": 1.0},  # Medium distance, original size
    "far": {"offset": 5, "scale": 0.8}       # Far distance, reduce the person size
}

# Scale and adjust the person image to enhance the depth illusion
def preprocess_person_image(person_image, scale):
    # The input is already a PIL.Image
    person_image_pil = person_image

    # Scale the person image based on the scale parameter
    new_size = (int(person_image_pil.width * scale), int(person_image_pil.height * scale))
    scaled_person = person_image_pil.resize(new_size, Image.LANCZOS)

    # Further adjust brightness or contrast to match the background
    enhancer = ImageEnhance.Brightness(scaled_person)
    scaled_person = enhancer.enhance(1.1)

    return scaled_person  # Return a PIL.Image object

# Insert the person image based on depth, adding position parameter
def insert_person_with_depth(left_image, right_image, person_image, depth="close", position=(50, 50)):
    config = depth_configs.get(depth, depth_configs["close"])
    offset = config["offset"]
    scale = config["scale"]

    # Preprocess the person image: scale and adjust brightness/contrast
    processed_person_image = preprocess_person_image(person_image, scale)

    # Insert the processed image
    return insert_person(left_image, right_image, processed_person_image, depth, position)

# Verify insertion results at different depth levels
if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Load the input image as a PIL.Image object
    input_image = Image.open("input_image.jpg")  # Replace with your image file path

    # Perform segmentation
    person_image = segment_person(input_image, model)

    # Load and split the stereo image
    left_image, right_image = load_stereo_image("sbs_neu.jpg")

    # For each depth level, insert and save the results, example position is (50, 50)
    for depth in ["close", "medium", "far"]:
        left_with_person, right_with_person = insert_person_with_depth(
            left_image, right_image, person_image, depth, position=(50, 50)
        )

        # Save files with depth level in the filename
        left_with_person.save(f"left_with_person_{depth}.png")
        right_with_person.save(f"right_with_person_{depth}.png")
        print(f"{depth.capitalize()} depth level images saved as left_with_person_{depth}.png and right_with_person_{depth}.png")