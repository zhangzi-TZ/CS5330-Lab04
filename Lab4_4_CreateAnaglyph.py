import numpy as np
from PIL import Image
from Lab4_1_SegmentImage import segment_person, load_model
from Lab4_2_InsertIntoStereoImages import load_stereo_image
from Lab4_3_AdjustDepth import insert_person_with_depth


# Create an Anaglyph image
def create_anaglyph_image(left_image, right_image):
    # Convert images to RGB format (remove alpha channel)
    left_image = left_image.convert("RGB")
    right_image = right_image.convert("RGB")

    # Convert images to NumPy arrays
    left_np = np.array(left_image)
    right_np = np.array(right_image)

    # Create an empty Anaglyph image
    anaglyph_np = np.zeros_like(left_np)

    # Combine images
    anaglyph_np[..., 0] = left_np[..., 0]  # Red channel from the left image
    anaglyph_np[..., 1] = right_np[..., 1]  # Green channel from the right image
    anaglyph_np[..., 2] = right_np[..., 2]  # Blue channel from the right image

    # Convert back to PIL.Image and return
    anaglyph_image = Image.fromarray(anaglyph_np)
    return anaglyph_image


# Verify the result of creating a 3D image
if __name__ == "__main__":
    # Load the model and segment the image
    model = load_model()

    # Load the input image as a PIL.Image object
    input_image = Image.open("input_image.jpg")

    # Perform segmentation
    person_image = segment_person(input_image, model)

    # Load and split the stereo image
    left_image, right_image = load_stereo_image("sbs_neu.jpg")

    # Generate and save Anaglyph images for each depth level
    for depth in ["close", "medium", "far"]:
        # Insert the person image into left and right images at different depths
        left_with_person, right_with_person = insert_person_with_depth(
            left_image, right_image, person_image, depth
        )

        # Ensure the inserted images are not None
        if left_with_person is None or right_with_person is None:
            print(f"Insertion failed at {depth} depth level. Check left_with_person and right_with_person.")
            continue  # Skip to the next depth level

        # Create Anaglyph image
        anaglyph_image = create_anaglyph_image(left_with_person, right_with_person)

        # Save the Anaglyph image with depth level in the filename
        anaglyph_image.save(f"anaglyph_image_{depth}.png")
        print(f"{depth.capitalize()} depth level Anaglyph image saved as anaglyph_image_{depth}.png")