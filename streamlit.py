# import streamlit as st
# import requests

# API_URL = "http://localhost:8000"  # Change to your FastAPI server location

# st.set_page_config(page_title="Nephele 3.0 Teaching Assistant", page_icon="🎤", layout="centered")

# st.title("🎤 Nephele 3.0 Teaching Assistant")
# st.markdown("Upload a lesson document or audio and interact with your AI Teaching Assistant.")

# with st.expander("Step 1: Load Lesson Document (URL or PDF)"):
#     source = st.text_input("Document Source (URL or PDF Path):")
#     if st.button("Load Document") and source:
#         resp = requests.post(f"{API_URL}/load_document/", data={"source": source})
#         if resp.ok and resp.json().get("success"):
#             st.success("Document loaded successfully!")
#         else:
#             st.error("Failed to load document.")

# with st.expander("Step 2: Teach Lesson (Audio Upload)"):
#     uploaded_file = st.file_uploader("Upload topic audio (.wav/.mp3)", type=["wav", "mp3"])
#     if st.button("Teach Lesson") and uploaded_file:
#         files = {"file": (uploaded_file.name, uploaded_file, "audio/mpeg")}
#         resp = requests.post(f"{API_URL}/teach_lesson/", files=files)
#         if resp.ok:
#             data = resp.json()
#             st.write(f"**Transcribed Topic:** {data['topic']}")
#             st.write(f"**Lesson:** {data['lesson']}")
#         else:
#             st.error("Teaching lesson failed.")

# with st.expander("Step 3: Ask a Doubt (Audio Upload)"):
#     doubt_file = st.file_uploader("Ask a doubt (.wav/.mp3)", type=["wav", "mp3"], key="doubt")
#     if st.button("Submit Doubt") and doubt_file:
#         files = {"file": (doubt_file.name, doubt_file, "audio/mpeg")}
#         resp = requests.post(f"{API_URL}/doubt/", files=files)
#         if resp.ok:
#             data = resp.json()
#             st.write(f"**Question:** {data['question']}")
#             st.write(f"**Related to Lesson:** {data['related']}")
#             st.write(f"**Answer:** {data['answer']}")
#             if data["audio_file"]:
#                 st.audio(f"{API_URL}/audio_files/{data['audio_file']}")
#         else:
#             st.error("Doubt handling failed.")

# if st.button("Cleanup Audio Files"):
#     resp = requests.post(f"{API_URL}/cleanup_audio_files/")
#     if resp.ok:
#         st.success(f"Cleaned: {', '.join(resp.json().get('deleted_files', []))}")
#     else:
#         st.error("Cleanup failed.")

# st.write("---")
# st.info("All features are voice-enabled. Make sure your FastAPI backend is running and accessible.")
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Nephele 3.0 Teaching Assistant", page_icon="🎤", layout="centered")
st.title("🎤 Nephele 3.0 Teaching Assistant")
st.markdown("Upload a lesson document or audio and interact with your AI Teaching Assistant.")

with st.expander("Step 1: Load Lesson Document (URL or PDF)"):
    source = st.text_input("Document Source (URL or PDF Path):")
    if st.button("Load Document") and source:
        resp = requests.post(f"{API_URL}/load_document/", data={"source": source})
        if resp.ok and resp.json().get("success"):
            st.success("Document loaded successfully!")
        else:
            st.error("Failed to load document.")

with st.expander("Step 2: Teach Lesson (Record Audio)"):
    audio_file = st.audio_input("🎤 Record topic audio")
    if audio_file and st.button("Teach Lesson"):
        files = {"file": ("topic.wav", audio_file, "audio/wav")}
        resp = requests.post(f"{API_URL}/teach_lesson/", files=files)
        if resp.ok:
            data = resp.json()
            st.write(f"**Transcribed Topic:** {data['topic']}")
            st.write(f"**Lesson:** {data['lesson']}")
        else:
            st.error("Teaching lesson failed.")

with st.expander("Step 3: Ask a Doubt (Record Audio)"):
    doubt_audio = st.audio_input("🎤 Ask your doubt by voice", key="doubt")
    if doubt_audio and st.button("Submit Doubt"):
        files = {"file": ("doubt.wav", doubt_audio, "audio/wav")}
        resp = requests.post(f"{API_URL}/doubt/", files=files)
        if resp.ok:
            data = resp.json()
            st.write(f"**Question:** {data['question']}")
            st.write(f"**Related to Lesson:** {data['related']}")
            st.write(f"**Answer:** {data['answer']}")
            if data["audio_file"]:
                st.audio(f"{API_URL}/audio_files/{data['audio_file']}")
        else:
            st.error("Doubt handling failed.")

if st.button("Cleanup Audio Files"):
    resp = requests.post(f"{API_URL}/cleanup_audio_files/")
    if resp.ok:
        st.success(f"Cleaned: {', '.join(resp.json().get('deleted_files', []))}")
    else:
        st.error("Cleanup failed.")