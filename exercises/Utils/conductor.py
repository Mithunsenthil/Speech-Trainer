import os
from pydub import AudioSegment
import whisper
import random

def save_uploaded_file(audio_file_path, destination_path):
    with open(audio_file_path, 'rb') as in_file:
        data = in_file.read()
    with open(destination_path, 'wb') as out_file:
        out_file.write(data)

def convert_wav_to_mp3(wav_file, mp3_file="output.mp3"):
    try:
        audio = AudioSegment.from_wav(wav_file)
        audio.export(mp3_file, format="mp3")
        print(f"Converted {wav_file} to {mp3_file}")
        return mp3_file
    except Exception as e:
        print("Error converting WAV to MP3:", e)
        raise

def transcribe_audio(mp3_file):
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Transcribing audio...")
    result = model.transcribe(mp3_file)
    return result["text"]

def process_audio_upload(audio_file_path):
    wav_path = "temp_output.wav"
    mp3_path = "temp_output.mp3"
    try:
        print("Converting webm to wav...")
        audio_segment = AudioSegment.from_file(audio_file_path, format="webm")
        audio_segment.export(wav_path, format="wav")
    except Exception as e:
        print("Error converting webm to wav:", e)
        raise

    convert_wav_to_mp3(wav_path, mp3_path)
    transcript = transcribe_audio(mp3_path)
    for file in [wav_path, mp3_path]:
        if os.path.exists(file):
            os.remove(file)
    return transcript

def process_conductor_audio(audio_file_path):
    """
    Process the Conductor exercise audio:
      1. Transcribe the audio.
      2. Simulate real-time analysis:
         - Track energy levels (simulated by a score)
         - Analyze vocal variety (simulated score)
         - Provide mood matching feedback (pass/fail or score)
         - Generate personalized improvement suggestions (from a set of suggestions)
    """
    transcript = process_audio_upload(audio_file_path)
    
    # Simulate analysis scores:
    energy_level = round(random.uniform(0, 10), 2)
    vocal_variety = round(random.uniform(0, 10), 2)
    mood_match_score = round(random.uniform(0, 10), 2)
    
    # Generate personalized improvement suggestions (simulated)
    suggestions_list = [
        "Practice modulating your pitch.",
        "Vary your pace for better emphasis.",
        "Focus on clear articulation.",
        "Experiment with different tones."
    ]
    improvement_suggestions = random.sample(suggestions_list, 2)
    
    result = {
        "transcript": transcript,
        "energy_level": energy_level,
        "vocal_variety": vocal_variety,
        "mood_match_score": mood_match_score,
        "improvement_suggestions": improvement_suggestions
    }
    return result
