# app.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

# ————————————————————————————————
# GANTI dengan token kau:
ACCESS_TOKEN = "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"
# Nama repo Spaces nanti: Fadhil04/Earl_Coder_Ai
# ————————————————————————————————

MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"

# Load tokenizer & model dengan token hard-coded
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=ACCESS_TOKEN
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=ACCESS_TOKEN
)

def generate_code(prompt: str, max_new_tokens: int = 256) -> str:
    """Terima prompt teks, hasilkan kod."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Sediakan interface Gradio
demo = gr.Interface(
    fn=generate_code,
    inputs=[
        gr.Textbox(lines=3, label="💬 Prompt Anda"),
        gr.Slider(minimum=64, maximum=4096, step=64, value=256, label="🔢 Max New Tokens")
    ],
    outputs=gr.Textbox(label="🖥️ Hasil Kod"),
    title="Earl_Coder_Ai (DeepSeek Coder 6.7B)",
    description=(
        "AI penjana kod open-source berdasarkan DeepSeek Coder 6.7B. "
        "Masukkan prompt, dan dapatkan kod Python/JavaScript/…"
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
