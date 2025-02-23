import os
from pydub import AudioSegment
import whisper

def analyze_transcript(transcript: str, response_time: float = 1) -> dict:
    # Response Timing Score: simple thresholds (example values)
    if response_time <= 3:
        response_timing_score = 10
    elif response_time <= 5:
        response_timing_score = 8
    elif response_time <= 7:
        response_timing_score = 6
    else:
        response_timing_score = 4

    # Speech Continuity Score: count filler words
    filler_words = ['um', 'uh', 'like', 'you know', 'er']
    transcript_lower = transcript.lower()
    filler_count = sum(transcript_lower.count(word) for word in filler_words)
    speech_continuity_score = max(0, 10 - filler_count)

    # Analogy Relevance: non-empty transcript gets full score
    relevance_score = 10 if transcript.strip() != "" else 0

    # Creativity Score: lexical diversity (unique words / total words)
    words = transcript.split()
    total_words = len(words)
    unique_words = len(set(words))
    lexical_diversity = (unique_words / total_words) if total_words > 0 else 0
    creativity_score = lexical_diversity * 10

    overall_score = (
        response_timing_score +
        speech_continuity_score +
        relevance_score +
        creativity_score
    ) / 4

    return {
        "response_timing_score": response_timing_score,
        "speech_continuity_score": speech_continuity_score,
        "analogy_relevance_score": relevance_score,
        "creativity_score": creativity_score,
        "overall_score": overall_score,
    }

def save_uploaded_file(audio_file_path, destination_path):
    """
    Reads the content of a file at 'audio_file_path' and writes it to 'destination_path'.
    This function assumes that audio_file_path is a string representing a valid file path.
    """
    with open(audio_file_path, 'rb') as in_file:
        data = in_file.read()
    with open(destination_path, 'wb') as out_file:
        out_file.write(data)

def convert_wav_to_mp3(wav_file, mp3_file="output.mp3"):
    """
    Convert a WAV file to MP3 using pydub.
    Requires ffmpeg to be installed on the system.
    """
    try:
        audio = AudioSegment.from_wav(wav_file)
        audio.export(mp3_file, format="mp3")
        print(f"Converted {wav_file} to {mp3_file}")
        return mp3_file
    except Exception as e:
        print("Error converting WAV to MP3:", e)
        raise

def transcribe_audio(mp3_file):
    """
    Transcribe the provided MP3 file using Whisper.
    """
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Transcribing audio...")
    result = model.transcribe(mp3_file)
    return result["text"]

def process_audio_upload(audio_file_path):
    """
    Process the uploaded audio file:
      1. Convert the uploaded webm file to WAV.
      2. Convert the WAV file to MP3.
      3. Transcribe the MP3 file using Whisper.
      
    Note: In this example we assume the uploaded file is in webm format.
    """
    # Define temporary file paths
    wav_path = "temp_output.wav"
    mp3_path = "temp_output.mp3"

    try:
        # Convert the webm file to WAV.
        # pydub can usually detect the file type from the file extension.
        print("Converting webm to wav...")
        audio_segment = AudioSegment.from_file(audio_file_path, format="webm")
        audio_segment.export(wav_path, format="wav")
    except Exception as e:
        print("Error converting webm to wav:", e)
        raise

    # Convert WAV to MP3
    convert_wav_to_mp3(wav_path, mp3_path)
    
    # Transcribe the MP3 file using Whisper
    transcript = transcribe_audio(mp3_path)
    
    # Clean up temporary files
    for file in [wav_path, mp3_path]:
        if os.path.exists(file):
            os.remove(file)
    
    return transcript
