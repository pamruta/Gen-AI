import spaces

import torch
import gradio as gr

from diffusers import StableDiffusionXLPipeline

# specify which stable-diffusion model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# load model
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe = pipe.to("cuda")


# design room decor as per the selected style and color theme
@spaces.GPU(duration=60)
def design(room_style, room_type, color_scheme):

  # describe things to avoid during generation process
  negative_prompt = "blurry, distorted, flashy bright gaudy colors, cartoon, sketch"

  # generate text description
  prompt = "photorealistic, high quality " + room_style + " style " + room_type + ", using soft color tones and shades of " + color_scheme + " and white"

  return pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    strength = 0.75,
    width=512,
    height=512
  ).images[0]

# css style
css = """
button.primary {
    background-color: #4682b4 !important;
    border-color: #c0c0c0 !important;
    color: #ffffff !important;
}

h1, h2 {
    font-family: cursive !important;
    color: #696969 !important;
}
"""

# user-interface

image_gallery = []
for k in range(1, 17):
    image_gallery.append("gallery/" + str(k) + ".jpg")

sample_image = "sample.jpg"
with gr.Blocks(css=css) as demo:
    gr.Interface(
        fn=design,
        inputs=[
            gr.Dropdown(
                choices=["Bohemian", "Eclectic", "Modern", "Nordic", "Scandinavian"],
                value="Bohemian",
                label="Room Style"),
            gr.Dropdown(
                choices=["Living Room", "Bedroom", "Kitchen", "Office", "Bathroom"],
                value="Living Room",
                label="Room Type"),
            gr.Dropdown(
                choices=["Blue", "Peach", "Plum", "Purple", "Turquoise", "Brown", "Grey"],
                value="Blue",
                label="Color Choice")
        ],
        outputs=gr.Image(label="room design", value=sample_image),
        title="Room Decor",
        flagging_mode='never',
    )

    gr.Markdown("## Design Gallery ")
    gr.Gallery(value = image_gallery, rows=3, columns = 4, 
        label = "Gallery", height="auto")

if __name__ == "__main__":
    demo.launch()