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


def load_text_model(model_dict,
                    n_ctx=512,
                    n_batch=126,
                    chat_handler=None) -> Llama:
    
    model = Llama(model_path=model_dict["path"],
                  n_ctx=n_ctx,
                  n_batch=n_batch,
                  n_gpu_layers=-1,
                  chat_handler=chat_handler)
    return model


def generate_response_text(model,
                           prompt="Who is the CEO of Apple?",
                           max_tokens=256,
                           temperature=0.1,
                           top_p=0.5,
                           echo=False,
                           stop=["#"],):
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


def load_image_model(model_dict: dict) -> Llama:
    if model_dict["name"] == "llava":
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_handler = Llava15ChatHandler(clip_model_path=model_dict["clip_model_path"])
        model = Llama(
                model_path=model_dict["path"],
                chat_handler=chat_handler,
                n_ctx=2048, # n_ctx should be increased to accommodate the image embedding
                n_gpu_layers=-1)
        return model


def generate_response_image(model: Llama, image_path: str):
    response = model.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    {"type" : "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": image_path} }
                ]
            }
        ]
    )
    return response["choices"][0]["message"]["content"]
