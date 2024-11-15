import gradio as gr
from PIL import Image
from Lab4_1_SegmentImage import segment_person, load_model
from Lab4_2_InsertIntoStereoImages import load_stereo_image
from Lab4_3_AdjustDepth import insert_person_with_depth
from Lab4_4_CreateAnaglyph import create_anaglyph_image

# Define the processing function, adding x_position and y_position parameters
def generate_anaglyph(input_image, depth, x_position, y_position, background_image=None):
    # Load the model and segment the person image
    model = load_model()
    person_image = segment_person(input_image, model)

    # Use default background if none is uploaded
    if background_image is None:
        background_image = "sbs_neu.jpg"
        left_image, right_image = load_stereo_image(background_image)
    else:
        left_image, right_image = load_stereo_image(background_image)

    # Insert the person image, passing position parameters
    left_with_person, right_with_person = insert_person_with_depth(
        left_image, right_image, person_image, depth, position=(x_position, y_position)
    )

    # Create Anaglyph image
    anaglyph_image = create_anaglyph_image(left_with_person, right_with_person)

    return anaglyph_image

# Create the Gradio interface, adding sliders to adjust position
iface = gr.Interface(
    fn=generate_anaglyph,
    inputs=[
        gr.Image(type="pil", label="Upload an image containing a person"),
        gr.Radio(["close", "medium", "far"], label="Select depth level", value="close"),
        gr.Slider(0, 100, value=50, label="Horizontal position (0-100)"),
        gr.Slider(0, 100, value=50, label="Vertical position (0-100)"),
        gr.Image(type="pil", label="Upload background image (optional)")
    ],
    outputs=gr.Image(type="pil", label="Generated Anaglyph Image"),
    title="3D Anaglyph Image Generator",
    description="Upload an image containing a person, select depth level (close, medium, far), adjust the person's position in the scene (horizontal and vertical), and optionally upload a custom background to generate an Anaglyph 3D image.",
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(share= True)