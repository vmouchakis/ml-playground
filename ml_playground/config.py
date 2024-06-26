from pydantic import BaseModel


class Modalities(BaseModel):
    text: str = "text"
    image: str = "image"
    audio: str = "audio"


class ModelTasks(BaseModel):
    text_generation: list = ["zephyr"]
    code_generation: list = ["codellama"]


models_dict = {
    "text": {
        "zephyr": {
            "name": "zephyr",
            "path": "models/zephyr-7b-beta.Q4_0.gguf",
            "tasks": ["text generation"],
            "type": "text",
            "prompt_template": """<|im_start|>system
                You are a helpful chatbot.<|im_end|>
                <|im_start|>user
                {input}<|im_end|>""",
            "supported": True
        },
        "codellama": {
            "name": "codellama",
            "path": "models/codellama-7b.Q8_0.gguf",
            "tasks": ["code generation", "code completion"],
            "type": "code",
            "prompt_template": None,
            "supported": True
        },
    },
    "image": {
        "llava": {
            "name": "llava",
            "path": "models/ggml_llava-v1.5-7b-q5_k.gguf",
            "clip_model_path": "models/mmproj-model-f16.gguf",
            "tasks": ["image description", "visual question answering"],
            "type": "image",
            "prompt_template": {
                "messages": [
                    {"role": "system", "content": "You are an assistant who perfectly describes images."},
                    {"role": "user", "content": [{"type": "text", "text": "What's in this image?"}, {"type": "image_url", "image_url": {"url": "{image_path}"}}]}
                ]
            },
            "supported": True
        },
    },
    "audio": {
        "whisper": {
            "name": "whisper",
            "path": "models/ggml-whisper-large-v3.bin",
            "tasks": ["speech-to-text"],
            "type": "audio",
            "prompt_templae": None,
            "supported": False
        }
    }
}

def get_supported_model_types(models_dict):
    supported_keys = []
    for key, models in models_dict.items():
        if any(model_info.get('supported', False)
               for model_info in models.values()):
            supported_keys.append(key)
    return supported_keys
