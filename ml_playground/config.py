import os

clip_model = "models/mmproj-model-f16.gguf"

models_dict = {
    "zephyr": "models/zephyr-7b-beta.Q4_0.gguf",
    "codellama": "models/codellama-7b.Q8_0.gguf",
    "llava": "models/ggml_llava-v1.5-7b-q5_k.gguf",
}