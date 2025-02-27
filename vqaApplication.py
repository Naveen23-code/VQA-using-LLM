
import streamlit as st # type: ignore
from PIL import Image
import requests
from io import BytesIO
import google.generativeai as genai

# Function to load image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the image from URL: {e}")
        return None
    except IOError as e:
        st.error(f"Error opening the image file: {e}")
        return None

# Function to process image and ask VQA questions
def process_image(image, questions, api_key):
    # Configure the generative AI client
    genai.configure(api_key=api_key)
    # Instantiate the Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    responses = []
    for question in questions:
        response = model.generate_content([question, image])
        responses.append((question, response.text))
    return responses

# Streamlit app
def main():
    st.title("Visual Question Answering with Gemini")
    
    # API Key Input
    api_key = st.text_input("Enter your API key:", type="password")
    
    if not api_key:
        st.warning("Please enter your API key.")
        return

    # Image upload section
    image_source = st.radio("Select image source", ('URL', 'Local Upload'))
    
    img = None  # Initialize img variable

    if image_source == 'URL':
        image_url = st.text_input("Enter the URL of the image:")
        if image_url:
            img = load_image_from_url(image_url)
            if img:
                st.image(img, caption='Uploaded Image', use_column_width=True)
    
    elif image_source == 'Local Upload':
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image', use_column_width=True)
            except IOError as e:
                st.error(f"Error opening the uploaded file: {e}")

    # Questions input section
    questions = st.text_area("Enter your questions (one per line):", height=200)
    questions_list = questions.splitlines()
    
    if st.button("Submit Questions"):
        if img and questions_list:
            responses = process_image(img, questions_list, api_key)
            for question, answer in responses:
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {answer}")
        else:
            if not img:
                st.error("Please upload a valid image or provide a valid URL.")
            if not questions_list:
                st.error("Please enter your questions.")

if __name__ == "__main__":
    main()
# Use Ctrl + C to stop the Application