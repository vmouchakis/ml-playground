import os

clip_model = "models/mmproj-model-f16.gguf"

models_dict = {
    "text": {
        "zephyr": {
            "model_path":"models/zephyr-7b-beta.Q4_0.gguf"
        },
    },
    "code": {
        "codellama": {
            "model_path": "models/codellama-7b.Q8_0.gguf"
        },
    },
    "multimodal": {
        "llava": {
            "model_path": "models/ggml_llava-v1.5-7b-q5_k.gguf"
        },
    }
}