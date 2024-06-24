import streamlit as st
from config import models_dict
from utils import load_model, generate_text


def main():
    st.set_page_config(page_title="LeGo", page_icon="🤖", layout="wide")

    st.title("LeGo Model Playground")
    
    model_name = st.selectbox("What is the model of your choice?",
                              options=list(models_dict.keys()),
                              index=0)

    if model_name:
        model_path = models_dict[model_name]
        st.write(f"You have selected the model: {model_name}")
        st.write(f"Model path: {model_path}")
    

    question = st.text_area("Enter question: ", height=150)

    if question:
        model = load_model(model_path)
        st.write(f"You asked: {question}")
        answer = generate_text(model=model, prompt=question)
        st.write(f"Answer:\n{answer}")



if __name__ == "__main__":
    main()
