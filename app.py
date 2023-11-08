import streamlit as st
from PIL import Image
import base64
import io
import os
from get_best_matching import get_best_matching


# Load your images
root_folder = os.path.dirname(os.path.abspath(__file__))
cow_images_dir = os.path.join(root_folder,"ten_cow1")
images_path = [os.path.join(cow_images_dir,i) for i in os.listdir(cow_images_dir)]
images = [Image.open(i) for i in images_path]


# Function to convert image to bytes
def get_image_bytes(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

# Function to process the image
def process_image(img):
    # Replace this function with your actual ML model logic
    # This is just a placeholder
    process_file_name = 'process_img.png'
    img.save(process_file_name)
    result = get_best_matching(process_file_name)
    img1_byte  = get_image_bytes(Image.open(process_file_name))
    img2_byte = get_image_bytes(Image.open(result[0]))
    img3_byte = get_image_bytes(Image.open(result[1]))
    if os.path.exists(process_file_name):
        os.remove(process_file_name)
    if os.path.exists(r"matched_result.jpg"):
        os.remove("matched_result.jpg")
    return img1_byte, img2_byte, img3_byte, result[2]

# Create a dictionary to hold the image name and bytes
image_dict = {f"Image {i}": get_image_bytes(images[i-1]) for i in range(1, 11)}

# for i in range(0, 10, 5):
#     cols = st.columns(5)
#     for j in range(5):
#         image = Image.open(io.BytesIO(image_dict[f"Image {i+j+1}"]))
#         cols[j].image(image, width=125)

uploaded_file = st.file_uploader("Choose an image from your desktop", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # Convert the uploaded file to an image
    uploaded_image = Image.open(uploaded_file)
    selected_image = uploaded_image
else:
    # Create a selectbox for the images
    selected_image_name = st.selectbox('Please select an image:', list(image_dict.keys()))

    # Get the image bytes of the selected image
    image_bytes = image_dict[selected_image_name]

    # Convert bytes to image
    selected_image = Image.open(io.BytesIO(image_bytes))
if st.button('Process Image'):
    with st.spinner("Processing..."):
        # Process the image
        print("Processing with Image1")
        img1, img2, img3, text = process_image(selected_image) 
        # Display the results
        st.header("Generated HASH:-")
        st.write(text)
        st.image(img3, caption='Matching Descriptor')
        st.image(img1, caption='Selected Image',width=300)
        st.image(img2, caption='Image matched with',width=300)
