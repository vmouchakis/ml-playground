import os
import base64
import streamlit as st
from config import models_dict
from utils import load_model, generate_text, get_llava_response


def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary directory and return the file path."""
    temp_dir = "temp_uploaded_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def image_to_base64_with_prefix(local_path):
    with open(local_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"


def main():
    st.set_page_config(page_title="LeGo", page_icon="🤖", layout="wide")

    st.title("LeGo Model Playground")

    model_type = st.selectbox("What is the model type of your choice?",
                              options=list(models_dict.keys()),
                              index=None)
    if model_type and model_type == 'text':
        model_name = st.selectbox("What is the model of your choice?",
                                options=list(models_dict[model_type].keys()),
                                index=None)

        if model_name:
            model_path = models_dict[model_type][model_name]["model_path"]
            st.write(f"You have selected the model: {model_name}")
            st.write(f"Model path: {model_path}")

        question = st.text_area("Enter question: ", height=150)

        if question:
            model = load_model(model_path)
            st.write(f"You asked: {question}")
            answer = generate_text(model=model, prompt=question)
            st.write(f"Answer:\n{answer}")

    elif model_type and model_type == 'multimodal':
        model_name = st.selectbox("What is the model of your choice?",
                                options=list(models_dict[model_type].keys()),
                                index=None)
        if model_name and model_name.lower() == "llava":
                image_file = st.file_uploader("Upload an image for the Lava model", type=["jpg", "jpeg", "png"])
                
                if image_file:
                    st.image(image_file, caption="Uploaded Image", use_column_width=True)
                    image_path = save_uploaded_file(image_file)
                    image_data = image_to_base64_with_prefix(image_path)
                    response = get_llava_response(image_path=image_data)
                    st.write(f"Description: {response}")



if __name__ == "__main__":
    main()
