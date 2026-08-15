!pip install -q -U transformers accelerate gradio

import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!")

def generate_text(topic, max_new_tokens=200):

    if not topic.strip():
        return "Please enter a topic."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a text generation assistant. "
                "Write one clear, coherent and informative paragraph "
                "about the user's topic. Stay focused on the topic. "
                "Do not invent unnecessary details."
            )
        },
        {
            "role": "user",
            "content": (
                f"Write a well-structured paragraph about: {topic}"
            )
        }
    ]

    # Convert conversation into model input
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only newly generated tokens
    generated_tokens = outputs[
        0
    ][inputs["input_ids"].shape[-1]:]

    # Convert tokens to text
    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return generated_text.strip()

topic = "Space exploration"

result = generate_text(topic)

print("Topic:")
print(topic)

print("\nGenerated Paragraph:")
print("-" * 80)
print(result)
print("-" * 80)

topics = [
    "Artificial Intelligence",
    "Space exploration",
    "Climate change",
    "Importance of education",
    "Web development",
    "Healthcare technology",
    "Renewable energy",
    "Indian space research"
]

for topic in topics:

    print("\n" + "=" * 80)
    print("TOPIC:", topic)
    print("=" * 80)

    result = generate_text(topic)

    print(result)

while True:

    topic = input(
        "\nEnter a topic (or type 'exit' to stop): "
    )

    if topic.lower().strip() == "exit":
        print("\nText generation stopped.")
        break

    result = generate_text(topic)

    print("\nGenerated Text:")
    print("-" * 80)
    print(result)
    print("-" * 80)

def generate_for_interface(topic):

    return generate_text(topic, max_new_tokens=200)


interface = gr.Interface(
    fn=generate_for_interface,
    inputs=gr.Textbox(
        label="Enter a Topic",
        placeholder="Example: Artificial Intelligence",
        lines=2
    ),
    outputs=gr.Textbox(
        label="Generated Paragraph",
        lines=10
    ),
    title="Generative Text Model",
    description=(
        "Enter a topic and the AI model will generate "
        "a coherent paragraph about it."
    ),
    examples=[
        ["Artificial Intelligence"],
        ["Space exploration"],
        ["Climate change"],
        ["Importance of education"],
        ["Web development"],
        ["Renewable energy"]
    ]
)

interface.launch()
