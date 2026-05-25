# PREMIUM NETFLIX × OPENAI APP.PY
# Copy-Paste Entire File

import gradio as gr
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# =====================================================
# LOAD DATABASE
# =====================================================

DATABASE_READY = (
    os.path.exists("vector_store/faiss_index.bin")
    and
    os.path.exists("vector_store/metadata.pkl")
)

if DATABASE_READY:

    index = faiss.read_index(
        "vector_store/faiss_index.bin"
    )

    with open(
        "vector_store/metadata.pkl",
        "rb"
    ) as f:

        metadata = pickle.load(f)

else:

    index = None
    metadata = None

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================================
# LOAD LLM
# =====================================================

hf_token = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token
)

# =====================================================
# AI SEARCH
# =====================================================

def entertainment_ai(message, history):

    if index is None:

        history.append(
            {
                "role": "assistant",
                "content":
                """
⚡ Database still loading.
Please wait 1–2 minutes
and refresh.
"""
            }
        )

        return history, ""

    if not message.strip():
        return history, ""

    query_embedding = model.encode(
        [message]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        5
    )

    retrieved_context = []
    retrieved_metadata = []

    for idx in indices[0]:

        doc = metadata[idx]

        retrieved_context.append(
            doc["context"]
        )

        retrieved_metadata.append(
            f"""
🎬 {doc['series']}
S{doc['season']}E{doc['episode']}
• {doc['title']}
"""
        )

    combined_context = (
        "\n\n---\n\n".join(
            retrieved_context
        )
    )

    messages = [

        {
            "role": "system",
            "content": """
You are CineMind AI,
an advanced entertainment
analysis assistant.
Rules:
- Answer ONLY using context
- Never invent plot details
- Explain naturally
- Sound intelligent
- Keep answers concise
"""
        },

        {
            "role": "user",
            "content":
            f"""
Question:
{message}
Retrieved Context:
{combined_context}
"""
        }
    ]

    try:

        response = (
            client.chat.completions.create(
                messages=messages,
                max_tokens=220,
                temperature=0.25
            )
        )

        ai_answer = (
            response
            .choices[0]
            .message.content
        )

    except Exception as e:

        ai_answer = (
            f"❌ Error: {str(e)}"
        )

    metadata_text = "\n".join(
        retrieved_metadata
    )

    final_response = f"""
{ai_answer}
━━━━━━━━━━━━━━━━━━━
**Retrieved Sources**
{metadata_text}
"""

    history.append(
    (message, final_response)
)

    return history, ""

# =====================================================
# PREMIUM CSS
# =====================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Orbitron:wght@600&display=swap');
body {
    background: #0d0d0d !important;
}
.gradio-container {
    background:
    radial-gradient(
        circle at top,
        #1b0f10 0%,
        #0d0d0d 45%,
        #050505 100%
    ) !important;
    color: white !important;
    font-family: 'Inter', sans-serif;
}
/* HERO */
.hero-title {
    text-align: center;
    font-size: 56px;
    font-weight: 700;
    letter-spacing: -2px;
    background:
    linear-gradient(
        90deg,
        #ff3b3b,
        #ff6a6a
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero-subtitle {
    text-align: center;
    color: #a0a0a0;
    font-size: 18px;
    margin-bottom: 25px;
}
/* CHAT */
.chat-window {
    border-radius: 30px !important;
    border:
    1px solid rgba(
        255,
        255,
        255,
        0.08
    ) !important;
    background:
    rgba(
        255,
        255,
        255,
        0.04
    ) !important;
    backdrop-filter:
    blur(20px);
    box-shadow:
    0 0 50px rgba(
        255,
        0,
        0,
        0.08
    );
}
/* INPUT */
textarea {
    background:
    rgba(
        255,
        255,
        255,
        0.06
    ) !important;
    color:
    white !important;
    border:
    1px solid rgba(
        255,
        255,
        255,
        0.08
    ) !important;
    border-radius:
    18px !important;
    padding:
    18px !important;
    font-size:
    16px !important;
}
/* BUTTON */
button {
    background:
    linear-gradient(
        135deg,
        #ff2d2d,
        #ff5e5e
    ) !important;
    color:
    white !important;
    border:
    none !important;
    border-radius:
    18px !important;
    font-weight:
    600 !important;
    transition:
    0.25s ease !important;
    box-shadow:
    0 0 20px rgba(
        255,
        0,
        0,
        0.25
    ) !important;
}
button:hover {
    transform:
    translateY(-2px);
    box-shadow:
    0 0 30px rgba(
        255,
        0,
        0,
        0.35
    ) !important;
}
footer {
    display: none !important;
}
"""

# =====================================================
# UI
# =====================================================

with gr.Blocks() as demo:

    gr.HTML(
        """
<div class="hero-title">
CineMind AI
</div>
<div class="hero-subtitle">
Netflix × OpenAI Entertainment Intelligence Engine
</div>
"""
    )

    chatbot = gr.Chatbot(
    height=600,
    elem_classes="chat-window"
)

    with gr.Row():

        user_input = gr.Textbox(
            placeholder=
            "Ask anything about TV shows, characters, plots...",
            lines=2,
            scale=8
        )

        send_button = gr.Button(
            "🎬 Analyze",
            scale=1
        )

    gr.Examples(
        examples=[
            ["Who is Ranko Zamani?"],
            ["Why does Reddington want Elizabeth?"],
            ["Who is Walter White?"],
            ["What is the Red Wedding?"],
            ["Who is Homelander?"],
            ["What happened to Rachel?"]
        ],
        inputs=user_input
    )

    send_button.click(
        entertainment_ai,
        inputs=[
            user_input,
            chatbot
        ],
        outputs=[
            chatbot,
            user_input
        ]
    )

    user_input.submit(
        entertainment_ai,
        inputs=[
            user_input,
            chatbot
        ],
        outputs=[
            chatbot,
            user_input
        ]
    )

demo.launch(
    css=custom_css,
    theme=gr.themes.Soft()
)
