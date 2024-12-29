import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

# Configure page
st.set_page_config(page_title="Audio Transcription App", page_icon="🎙️")

# Initialize Google Gemini API
def initialize_gemini():
    # Try to get API key from secrets first, then environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            st.error("Google API Key not found. Please set it in .streamlit/secrets.toml or as an environment variable GOOGLE_API_KEY")
            st.stop()
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def transcribe_audio(audio_file, model):
    """
    Transcribe the uploaded audio file using Google Gemini
    """
    try:
        # Create a temporary file to store the uploaded audio
        with NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = Path(tmp_file.name)

        # Upload to Gemini and get transcription
        uploaded_file = genai.upload_file(tmp_path)
        response = model.generate_content([uploaded_file, "Transcribe this audio clip"])
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return response.text
    
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")
        return None

def main():
    st.title("🎙️ Audio Transcription App")
    st.write("Upload a WAV file to get its transcript using Google Gemini")
    
    # Initialize Gemini model
    try:
        model = initialize_gemini()
    except Exception as e:
        st.error("Failed to initialize Gemini API. Please check your API key.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Add a transcribe button
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(uploaded_file, model)
                
                if transcript:
                    st.success("Transcription Complete!")
                    st.markdown("### Transcript")
                    st.write(transcript)
                    
                    # Add download button for transcript
                    st.download_button(
                        label="Download Transcript",
                        data=transcript,
                        file_name="transcript.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()