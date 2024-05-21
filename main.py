import streamlit as st
from PIL import Image
import io
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector
from openai import OpenAI
import numpy as np
import torch

# Initialize OpenAI client
client = OpenAI(api_key="sk-proj-0cD3oLBfTMTsUGnvkwNyT3BlbkFJKC4h3kc1hAwukFVC3JkQ")

# Title and description
st.title('Image Upload and Display App')
st.write("""
This app allows you to upload an image file, processes it, and displays the input and output images in a formatted way.
""")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "svg", "webp", "bmp"])

if uploaded_file is not None:
    # Read the uploaded file
    input_image = Image.open(uploaded_file)
    
    # Display the input image
    st.write("### Input Image")
    st.image(input_image, caption="Uploaded Image", use_column_width=True)
    
    # Save the uploaded image to a buffer
    buf = io.BytesIO()
    input_image.save(buf, format="PNG")
    buf.seek(0)

    #convert image to url
    def upload_image_to_imgbb(image_path, api_key):
        with open(image_path, 'rb') as file:
            response = requests.post(
                'https://api.imgbb.com/1/upload',
                data={'key': api_key},
                files={'image': file}
            )
        return response.json()['data']['url']

    api_key = 'c18450c786749500447fe5fc6f072418'

    
    # the url of the uploaded image

    image_url = upload_image_to_imgbb(uploaded_file.name, api_key)

    # Check doc strings for more information
    seg_net = TracerUniversalB7(device='cpu',
                batch_size=1)

    fba = FBAMatting(device='cpu',
                    input_tensor_size=2048,
                    batch_size=1)

    trimap = TrimapGenerator()

    preprocessing = PreprocessingStub()

    postprocessing = MattingMethod(matting_module=fba,
                                trimap_generator=trimap,
                                device='cpu')

    interface = Interface(pre_pipe=preprocessing,
                        post_pipe=postprocessing,
                        seg_pipe=seg_net)

    response = requests.get(image_url)
    response.raise_for_status()  # check for HTTP errors

    image = Image.open(BytesIO(response.content))


    cat_wo_bg = interface([image])[0]
    cat_wo_bg.save('2.png')

    image_path = '/content/2.png'

    url = upload_image_to_imgbb(image_path, api_key)
    print('Image URL:', url)
    from IPython.display import display, Image

    image1_url = url

    # Display Image
    display(Image(url=image1_url))

    image = load_image(input_image)
    image = openpose(image)

    # Define the prompt function
    def prompt_animal(image_buffer:io.BytesIO):
        import base64
        image_url = "data:image/png;base64,"+base64.b64encode(image_buffer.getbuffer()).decode()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type":"text", "text": "RESPOND IN 1 WORD. Imagine you're an animal sorter aiding in spiritual discovery. Based on the image provided, what animal resonates with the person's essence? Please offer a single-word response. This animal should embody traits or qualities that align with the individual's character or aspirations. Avoid animals with negative connotations. Example responses include dog, cat, tiger, lion, bear, fish, shark, deer. YOUR RESPONSE SHOULD BE 1 WORD, RESEMBLING ONE OF THESE ANIMALS"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                    ]
                }
            ],
            max_tokens=10,
        )
        return response.choices[0].message.content

    animal = prompt_animal(buf)
    
    # Load the ControlNet and other models
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    controlnet_conditioning_scale = 0.5
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
    )

    image = load_image(input_image)
    image = openpose(image)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt = f'{animal}, ultra realistic, NO HUMAN, Replace BUT NOT ADD OR DELETE the human with a {animal}. Some emotion'
    negative_prompt = 'medium quality, unrealistic, distortion, unreasonable lighting, sketches, human'

    images = pipe(
        prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale, num_inference_steps=30
    ).images

    # Process the first image (assuming only one image is generated)
    output_image = images[0]

    # Convert the image to a format that can be displayed by Streamlit
    if isinstance(output_image, np.ndarray):
        output_image = Image.fromarray(output_image)

    # Display the output image using Streamlit
    st.write("### Output Image")
    st.image(output_image, caption="Processed Image", use_column_width=True)

    # Provide a download link for the processed image
    st.write("### Download Processed Image")
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png"
    )
else:
    st.write("Please upload an image file to proceed.")
