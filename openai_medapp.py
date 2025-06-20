import streamlit as st
import validators
from urllib.parse import urlparse
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

# Streamlit UI Setup
st.set_page_config(page_title="Medical Summarizer", page_icon="🩺")
st.title("🩺 Medical Content Summarizer")
st.subheader("Summarize YouTube videos or trusted medical websites")

# Sidebar for API key and model settings
with st.sidebar:
    openai_api_key = st.text_input("🔑 OpenAI API Key", value="", type="password")
    selected_model = st.selectbox("Choose OpenAI Model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"], index=0)
    temperature = st.slider("Temperature (creativity)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens (length of response)", min_value=50, max_value=400, value=150)

# URL Input
input_url = st.text_input("Paste a YouTube or Medical Website URL:")

# Normalize YouTube URL
def normalize_youtube_url(url):
    if "youtu.be" in url:
        video_id = url.split("/")[-1]
        return f"https://www.youtube.com/watch?v={video_id}"
    parsed = urlparse(url)
    if parsed.hostname == "www.youtube.com" and parsed.path.startswith("/shorts"):
        return None  # Shorts not supported
    return url

# Get YouTube transcript
def get_youtube_transcript(video_id: str):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return [Document(page_content=text)]
    except Exception as e:
        raise RuntimeError(f"Failed to get transcript: {e}")

# Prompt template
initial_prompt_template = """
You are a helpful assistant. Summarize the following content clearly and concisely.
Focus on the key points, main idea, and any technical or medical insights that stand out.

Content: {text}
"""

# Summarization logic
if st.button("Summarize the Content"):
    if not openai_api_key.strip():
        st.error("Please enter your OpenAI API key.")
    elif not input_url.strip():
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Processing and summarizing..."):
                docs = []
                normalized_url = normalize_youtube_url(input_url)

                if "youtube.com" in input_url or "youtu.be" in input_url:
                    if not normalized_url:
                        st.error("YouTube Shorts are not supported.")
                        st.stop()
                    video_id = normalized_url.split("v=")[-1]
                    docs = get_youtube_transcript(video_id)

                else:
                    parsed = urlparse(input_url)
                    if not any(domain in parsed.netloc for domain in ["nih.gov", ".gov", ".edu"]):
                        st.error("Only trusted medical sites (.gov, .edu, nih.gov) are allowed.")
                        st.stop()

                    loader = UnstructuredURLLoader(
                        urls=[input_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        },
                    )
                    docs = loader.load()

                llm = ChatOpenAI(
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=openai_api_key
                )

                single_prompt = PromptTemplate.from_template(initial_prompt_template)
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=single_prompt,
                    verbose=True,
                )

                st.info(f"Using '{selected_model}' model with temperature={temperature}, max_tokens={max_tokens}.")
                output_summary = chain.run(docs)

                st.success("✅ Summary:")
                st.write(output_summary)

        except Exception as e:
            st.error("An error occurred while summarizing.")
            st.exception(e)
