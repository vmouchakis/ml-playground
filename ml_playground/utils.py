import torch
from llama_cpp import Llama
import os
import urllib.request


def get_device() -> str:
    """
    Returns the available device: 'cpu', 'cuda' (GPU), or 'mps' (Metal Performance Shaders for Apple Silicon).
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def download_file(file_link, filename):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:
        print("File already exists.")


def load_model(model_path):
    model = Llama(model_path=model_path,  n_ctx=512, n_batch=126, n_gpu_layers=-1)
    return model


def generate_prompt_from_template(input):
    chat_prompt_template = f"""<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
{input}<|im_end|>"""
    return chat_prompt_template



def generate_text(model,
                  prompt="Who is the CEO of Apple?",
                  max_tokens=256,
                  temperature=0.1,
                  top_p=0.5,
                  echo=False,
                  stop=["#"],):
    
    # prompt = generate_prompt_from_template(prompt)

    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text