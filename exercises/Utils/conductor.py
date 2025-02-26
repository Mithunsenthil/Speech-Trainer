import os
from pydub import AudioSegment
import whisper
from groq import Groq
import librosa
import numpy as np
import nltk
import joblib

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
    model = whisper.load_model("small.en")
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

def generate_conductor_exercise():
    """
    Generate instructions for an exercise to improve vocal variety and expression.
    The response from the model is expected to contain lines like:
    
        Energy Levels: <energy_level1>, <energy_level2>, <energy_level3>
        Moods: <mood1>, <mood2>, <mood3>
        Improvement Suggestions: <suggestion1>, <suggestion2>, <suggestion3>
        
    Returns:
        dict: {
            "energy_levels": [<str>, ...],
            "moods": [<str>, ...],
            "improvement_suggestions": [<str>, ...]
        }
    """
    client = Groq(api_key="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")
    

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI Assistant. Generate an exercise prompt for improving vocal variety and expression. "
                    "The exercise should guide users through different energy levels and moods, and include instructions for real-time voice analysis "
                    "to track energy levels, analyze vocal variety, provide instant feedback on mood matching, and generate personalized improvement suggestions. "
                    "Please output the result in the following format exactly:\n\n"
                    "Energy level should only be High, Medium, or Low.\n"
                    "Mood should only be Joy, Sadness, Fear, Anger, Surprise, Neutral, Disgust, or Shame.\n"
                    "Energy Levels: <energy_level1>, <energy_level2>, <energy_level3>\n"
                    "Moods: <mood1>, <mood2>, <mood3>\n"
                    "Improvement Suggestions: <suggestion1>, <suggestion2>, <suggestion3>\n\n"
                    "Note: Moods must be chosen only from the following values: joy, sadness, fear, anger, surprise, neutral, disgust, shame."
                )
            },
            {
                "role": "user",
                "content": "Generate an exercise prompt for improving vocal variety and expression."
            }
        ],
        temperature=0.5,
        frequency_penalty=0.0,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    
    response_content = ""
    for chunk in completion:
        # Append each delta's content (if present) to the response_content.
        delta_text = chunk.choices[0].delta.content or ""
        response_content += delta_text
        print(delta_text, end="")  # print for debugging
    
    energy_levels = []
    moods = []
    improvement_suggestions = []
    
    lines = response_content.splitlines()
    for line in lines:
        if line.lower().startswith("energy levels:"):
            levels_str = line.split(":", 1)[1].strip()
            energy_levels = [lvl.strip() for lvl in levels_str.split(",") if lvl.strip()]
        elif line.lower().startswith("moods:"):
            moods_str = line.split(":", 1)[1].strip()
            moods = [m.strip() for m in moods_str.split(",") if m.strip()]
        elif line.lower().startswith("improvement suggestions:"):
            suggestions_str = line.split(":", 1)[1].strip()
            improvement_suggestions = [s.strip() for s in suggestions_str.split(",") if s.strip()]
    
    return {
        "energy_levels": energy_levels,
        "moods": moods,
        "improvement_suggestions": improvement_suggestions
    }

def remove_adjacent_duplicates(seq):
    if not seq:
        return []
    result = [seq[0]]
    for item in seq[1:]:
        if item != result[-1]:
            result.append(item)
    return result


def score_sequence_match(audio_sequence, target_sequence):
    # Remove adjacent duplicates 
    audio_sequence = remove_adjacent_duplicates(audio_sequence)
    
    if not target_sequence or not audio_sequence:
        return 0.0
    
    # Convert categorical values to numerical
    energy_map = {"low": 0, "medium": 1, "high": 2}
    
    # Convert sequences to numerical values
    num_audio = [energy_map.get(level.lower(), 1) for level in audio_sequence]
    num_target = [energy_map.get(level.lower(), 1) for level in target_sequence]
    
    # This allows for partial matching and timing flexibility
    max_score = len(target_sequence)
    score = 0.0
    
    i, j = 0, 0
    while i < len(num_audio) and j < len(num_target):
        # Exact match
        if num_audio[i] == num_target[j]:
            score += 1.0
        # Close match (off by one level)
        elif abs(num_audio[i] - num_target[j]) == 1:
            score += 0.5
        
        # Move forward in sequences
        if i < len(num_audio) - 1 and j < len(num_target) - 1:
            # Determine which sequence to advance
            if num_audio[i+1] == num_target[j]:
                i += 1
            elif num_audio[i] == num_target[j+1]:
                j += 1
            else:
                i += 1
                j += 1
        else:
            i += 1
            j += 1
    
    # Normalize to 0-10 scale
    final_score = (score / max_score) * 10
    
    return max(round(final_score, 2),10)

def analyze_mood_matches(transcript, target_moods_list):

    allowed_moods = ["joy", "sadness", "fear", "anger", "surprise", "neutral", "disgust", "shame"]
    sentences = nltk.sent_tokenize(transcript)
    results = []
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(current_dir, "emotion_classifier_pipe_lr.pkl")
    
    # Verify file exists
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Classifier pipeline not found at: {pipeline_path}")
    
    # Load the classifier
    with open(pipeline_path, "rb") as pipeline_file:
        loaded_pipe_lr = joblib.load(pipeline_file)

    # Related mood groups for partial matching
    related_moods = {
        "joy": ["surprise"],
        "sadness": ["shame", "disgust"],
        "fear": ["surprise", "shame"],
        "anger": ["disgust"],
        "surprise": ["joy", "fear"],
        "neutral": [],
        "disgust": ["anger", "sadness"],
        "shame": ["sadness", "fear"]
    }
    
    total_score = 0.3
    total_sentences = 1
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        total_sentences += 1
        
        # Get emotion prediction for this sentence
        predicted_emotion = loaded_pipe_lr.predict([sentence])[0].lower()
        
        # If we have a target mood for this sentence
        if target_moods_list and i < len(target_moods_list):
            expected_mood = target_moods_list.lower()
            
            # Exact match
            if predicted_emotion in expected_mood:
                total_score += 1.0
            # Related mood (partial match)
            elif predicted_emotion in related_moods.get(expected_mood, []):
                total_score += 0.5
            # Check if expected mood is in the related moods of predicted emotion
            elif expected_mood in related_moods.get(predicted_emotion, []):
                total_score += 0.3
        
        results.append({
            "sentence": sentence,
            "predicted_mood": predicted_emotion
        })
    
    # Normalize score to 0-10 scale
    print("MOOD sequence", results)
    mood_score = (total_score/ total_sentences) * 10
    return max(round(mood_score, 2),10)

def get_energy_level_sequence(audio_path, sr=22050, segment_duration=1.0):
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute RMS energy over frames
    rms = librosa.feature.rms(y=y)[0]
    hop_length = 512  # default hop_length in librosa.feature.rms
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Divide audio into segments of 'segment_duration' seconds.
    max_time = times[-1]
    num_segments = int(np.ceil(max_time / segment_duration))
    segment_energies = []
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        indices = np.where((times >= start_time) & (times < end_time))[0]
        if len(indices) == 0:
            avg_energy = 0.0
        else:
            avg_energy = np.mean(rms[indices])
        segment_energies.append(avg_energy)
    
    segment_energies = np.array(segment_energies)
    
    # Define thresholds using quantiles
    low_threshold = np.quantile(segment_energies, 0.33)
    high_threshold = np.quantile(segment_energies, 0.66)
    
    # Map each segment's average energy to a category.
    energy_sequence = []
    for energy in segment_energies:
        if energy < low_threshold:
            energy_sequence.append("low")
        elif energy < high_threshold:
            energy_sequence.append("medium")
        else:
            energy_sequence.append("high")
    return energy_sequence


def generate_targeted_suggestions(energy_score, mood_score, audio_sequence, target_sequence):
    """
    Generate targeted improvement suggestions based on actual performance
    """
    suggestions = []
    
    # Energy-related suggestions
    if energy_score < 5.0:
        if len(set(audio_sequence)) <= 1:
            suggestions.append("Try using more varied energy levels - your delivery was mostly at one level")
        else:
            # Check if specific energy levels are missing
            audio_levels = set(level.lower() for level in audio_sequence)
            target_levels = set(level.lower() for level in target_sequence)
            missing_levels = target_levels - audio_levels
            
            if "high" in missing_levels:
                suggestions.append("Work on incorporating higher energy moments in your delivery")
            if "low" in missing_levels:
                suggestions.append("Practice including quieter, more intimate moments in your delivery")
    
    # Mood-related suggestions
    if mood_score < 5.0:
        suggestions.append("Focus on matching your vocal tone to the intended emotion")
        suggestions.append("Try exaggerating the emotional quality to make it more recognizable")
    
    # General suggestions if we don't have enough targeted ones
    if len(suggestions) < 2:
        general_suggestions = [
            "Record yourself and listen back to identify subtle mood inconsistencies",
            "Try mirroring professional speakers to develop better vocal variety",
            "Work on maintaining consistent volume while varying your pitch and pace"
        ]
        
        # Add general suggestions until we have at least 2
        while len(suggestions) < 2 and general_suggestions:
            suggestions.append(general_suggestions.pop(0))
    
    return suggestions[:2]  # Return top 2 suggestions

def process_conductor_audio(audio_file_path, instructions, energy_levels, moods):
    """
    Process the Conductor exercise audio with improved scoring logic
    """
    transcript = process_audio_upload(audio_file_path)
    
    # Get energy sequence from audio
    audio_sequence = get_energy_level_sequence(audio_file_path, segment_duration=1.0)
    
    # Calculate energy level score 
    energy_level_score = score_sequence_match(audio_sequence, energy_levels)
    
    # Calculate mood score 
    mood_match_score = analyze_mood_matches(transcript, moods)
    
    # Generate personalized improvement suggestions based on actual scores
    improvement_suggestions = generate_targeted_suggestions(
        energy_level_score, 
        mood_match_score, 
        audio_sequence, 
        energy_levels
    )
    
    # Calculate overall score 
    overall_conductor_score = (energy_level_score + mood_match_score) / 2
    
    result = {
        "transcript": transcript,
        "instructions": instructions,
        "energy_levels": energy_levels,
        "moods": moods,
        "energy_level_score": energy_level_score,
        "mood_match_score": mood_match_score,
        "overall_conductor_score": overall_conductor_score,
        "improvement_suggestions": improvement_suggestions
    }
    return result
