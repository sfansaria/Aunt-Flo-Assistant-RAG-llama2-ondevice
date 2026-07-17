"""
Streamlit frontend for Aunt Flo Assistant.
Talks to the FastAPI backend over HTTP/SSE instead of loading the model
directly — this is what lets the frontend and inference layer scale and
deploy independently.

Run:
    streamlit run app.py
"""
import json
import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Aunt Flo Assistant ��🤖", page_icon="🌸")
st.title("Aunt Flo Assistant 🌸")
st.caption("A friendly guide to menstrual & reproductive health. Not a substitute for medical advice.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s['source']}, p.{s['page']}")

if prompt := st.chat_input("Ask about periods, cycles, symptoms, and more..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        sources = []

        with requests.post(
            f"{API_URL}/chat",
            json={"query": prompt, "session_id": st.session_state.session_id},
            stream=True,
            timeout=120,
        ) as resp:
            event, data_lines = None, []
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None or raw_line == "":
                    if event == "sources" and data_lines:
                        try:
                            sources = json.loads("".join(data_lines).replace("'", '"'))
                        except Exception:
                            sources = []
                    elif data_lines:
                        full_text += "".join(data_lines)
                        placeholder.markdown(full_text + "▌")
                    event, data_lines = None, []
                    continue
                if raw_line.startswith("event:"):
                    event = raw_line.split(":", 1)[1].strip()
                elif raw_line.startswith("data:"):
                    value = raw_line.split(":", 1)[1]
                    if value.startswith(" "):
                        value = value[1:]
                    data_lines.append(value)
                   

        placeholder.markdown(full_text)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s['source']}, p.{s['page']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_text, "sources": sources}
    )

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Aunt Flo Assistant answers questions using a curated knowledge base "
        "via retrieval-augmented generation. Answers include source citations."
    )
    if st.button("Clear conversation"):
        requests.post(f"{API_URL}/reset", params={"session_id": st.session_state.session_id})
        st.session_state.messages = []
        st.rerun()
