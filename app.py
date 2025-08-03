from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

def generate_code(prompt, max_new_tokens=256, temperature=0.7, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate_code,
    inputs=[
        gr.Textbox(lines=3, label="Arahan Kod"),
        gr.Slider(16, 2048, step=16, value=256, label="Maksimum Token Baru"),
        gr.Slider(0.1, 2.0, step=0.1, value=0.7, label="Temperature"),
        gr.Slider(0.1, 1.0, step=0.05, value=0.95, label="Top-p"),
    ],
    outputs=gr.Textbox(label="Hasil Kod"),
    title="DeepSeek Coder 6.7B",
    description="Model AI penjana kod menggunakan DeepSeek 6.7B."
)

if __name__ == "__main__":
    demo.launch()
