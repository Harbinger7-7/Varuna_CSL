"""
===============================================================================
 Project: Varuna — Shipyard Compliance Assistant

 Description:
     Local LLM-powered assistant to evaluate engineering procurement queries
     based on regulatory frameworks like GFR, DPP, GeM SOP, BIS/ISO standards.

 LLM: Mistral-7B-Instruct (GGUF via llama.cpp)
 RAG: ChromaDB + all-MiniLM-L6-v2 embeddings
 Prompt: Rule-based strict audit prompt

 Developers:
     - Anand Raj
       B.Tech Artificial Intelligence and Data Science, 2025
       Rajagiri School of Engineering and Technology, Kochi

     - Kestelyn Sunil Jacob
       B.Tech Artificial Intelligence and Data Science, 2025
       Rajagiri School of Engineering and Technology, Kochi
===============================================================================
"""

import time, os, gradio as gr
from app import get_answer_stream, set_interrupt, _clr  # Stream + control

# Paths and settings
BASE_DIR      = "C:/Users/user/Desktop/Internship_Projects/LLM_Mistral_base"
AVATAR_STATIC = os.path.join(BASE_DIR, "logo.gif")
AVATAR_STREAM = os.path.join(BASE_DIR, "loading.gif")

# Handles user submission stream
def handle_submit(user_msg, history):
    _clr()
    history = history or []
    history.append({"role": "user", "content": user_msg})
    assistant_msg = {"role": "assistant", "content": ""}
    history.append(assistant_msg)

    yield {
        chatbot   : history,
        avatar_img: gr.update(value=AVATAR_STREAM),
        stats_box : "<span class='stats-bar'>Elapsed: 0.0s | Tokens: 0</span>",
        state     : history
    }

    assistant_msg["content"] = ""
    t0 = time.perf_counter()
    for chunk in get_answer_stream(user_msg):
        assistant_msg["content"] += chunk
        elapsed = time.perf_counter() - t0
        tokens  = len(assistant_msg["content"])
        yield {
            chatbot  : history,
            stats_box: f"<span class='stats-bar'>Elapsed: {elapsed:.1f}s | Tokens: {tokens}</span>",
            state    : history
        }

    yield {avatar_img: gr.update(value=AVATAR_STATIC)}

# Stop button handler
def handle_stop():
    set_interrupt()
    return {
        avatar_img: gr.update(value=AVATAR_STATIC),
        stats_box : "<span class='stats-bar'>Stopped</span>",
    }

# UI layout and styling
with gr.Blocks(css="""
  .gradio-container {
      background-color:#0f2c4c;
      background-image:
          radial-gradient(circle at 10% 20%, rgba(0,119,204,0.15) 6%, transparent 7%),
          radial-gradient(circle at 20% 80%, rgba(0,170,255,0.18) 8%, transparent 9%),
          radial-gradient(circle at 70% 30%, rgba(0,100,180,0.12) 8%, transparent 9%),
          radial-gradient(circle at 90% 60%, rgba(0,160,255,0.12) 6%, transparent 7%);
      background-size:cover;
      color:#d6e8ff;font-family:'Consolas',monospace;min-height:100vh;}
  .title {font-size:34px;font-weight:bold;text-align:center;margin:20px 0;color:#a6dfff;}
  .chatbot-container{background:#0b1e33;border:1px solid #f0a500;color:#d6e8ff;padding:10px;max-height:450px;overflow-y:auto;white-space:pre-wrap;}
  .input-box textarea{background:#0b1e33!important;color:#d6e8ff!important;border:1px solid #f0a500!important;}
  .stats-bar{font-size:14px;color:#ffd700;font-weight:bold;}
  .stop-btn{background:#f0a500;color:#1e2a38;font-weight:bold;border:none;padding:8px 25px;border-radius:6px;}
  .stop-btn:hover{background:#d18e00;}
  #avatar-image img{object-fit:contain;width:300px;height:300px;}
  footer{display:none!important;}
""") as demo:

    gr.Markdown("<div class='title'>⚓ VARUNA – CSL Assistant ⚓</div>")

    with gr.Row(equal_height=True):
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(type="messages", elem_classes="chatbot-container", height=450)
            user_input = gr.Textbox(
                placeholder="Type your shipyard rule question and press Enter…",
                show_label=False, lines=1, max_lines=4, elem_classes="input-box")
            stats_box = gr.Markdown("<span class='stats-bar'>Elapsed: 0.0s | Tokens: 0</span>")
            stop_btn  = gr.Button("Stop", elem_classes="stop-btn")

        with gr.Column(scale=3):
            avatar_img = gr.Image(
                value=AVATAR_STATIC, show_label=False, interactive=False,
                show_download_button=False, height=300, width=300, elem_id="avatar-image")

    state = gr.State([])

    user_input.submit(handle_submit,
                      inputs=[user_input, state],
                      outputs=[chatbot, stats_box, avatar_img, state],
                      show_progress=True)

    user_input.submit(lambda x: "", inputs=user_input, outputs=user_input)
    stop_btn.click(handle_stop, outputs=[avatar_img, stats_box])

demo.launch(share=False, debug=False)
