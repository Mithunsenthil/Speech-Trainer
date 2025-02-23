import os
from pydub import AudioSegment
import whisper
import random

def save_uploaded_file(audio_file_path, destination_path):
    """
    Reads the content of a file at 'audio_file_path' and writes it to 'destination_path'.
    """
    with open(audio_file_path, 'rb') as in_file:
        data = in_file.read()
    with open(destination_path, 'wb') as out_file:
        out_file.write(data)

def convert_wav_to_mp3(wav_file, mp3_file="output.mp3"):
    """Convert a WAV file to MP3 using pydub (requires ffmpeg)."""
    try:
        audio = AudioSegment.from_wav(wav_file)
        audio.export(mp3_file, format="mp3")
        print(f"Converted {wav_file} to {mp3_file}")
        return mp3_file
    except Exception as e:
        print("Error converting WAV to MP3:", e)
        raise

def transcribe_audio(mp3_file):
    """Transcribe the provided MP3 file using Whisper."""
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Transcribing audio...")
    result = model.transcribe(mp3_file)
    return result["text"]

def process_audio_upload(audio_file_path):
    """
    Process an uploaded audio file by converting from webm to wav to mp3 and then transcribing.
    Assumes the input file is in webm format.
    """
    wav_path = "temp_output.wav"
    mp3_path = "temp_output.mp3"
    
    # Convert from webm to wav using pydub:
    try:
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

def process_triple_step_audio(audio_file_path, main_topic, distractor_words):
    """
    Process the Triple Step exercise audio:
      1. Transcribe the audio.
      2. Simulate evaluation by generating random scores for:
         - Speech coherence
         - Topic adherence
         - Distraction handling
      3. Simulate generation of additional distractor words based on the main topic.
    """
    # Use the existing processing logic to obtain a transcript.
    transcript = process_audio_upload(audio_file_path)
    
    # Simulate AI evaluation scores:
    coherence_score = round(random.uniform(8, 10), 2)
    topic_adherence_score = round(random.uniform(7, 10), 2)
    distraction_handling_score = round(random.uniform(6, 10), 2)
    
    # Simulate generating extra distractor words based on the main topic.
    extra_distractors = ["unexpected", "tangent"]
    generated_distractors = distractor_words + extra_distractors
    
    result = {
        "main_topic": main_topic,
        "transcript": transcript,
        "coherence_score": coherence_score,
        "topic_adherence_score": topic_adherence_score,
        "distraction_handling_score": distraction_handling_score,
        "generated_distractors": generated_distractors
    }
    return result
