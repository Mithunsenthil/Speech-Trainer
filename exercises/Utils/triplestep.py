import os
from pydub import AudioSegment
import whisper
import random
from groq import Groq
import nltk
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import nltk
from groq import Groq  

def analyze_distractor_smoothness(transcript: str, distractor_words: list) -> float:
    """
    Analyze how smoothly distractor words are integrated in the transcript.
    
    For each sentence containing any distractor word (case-insensitive), compute the cosine similarity
    between that sentence and its adjacent sentences (previous and next). A higher similarity indicates smoother integration.
    
    Returns a smoothness score between 0 and 10. If no sentence contains a distractor word, returns 10.0.
    """
    sentences = sent_tokenize(transcript)
    if not sentences:
        return 0.0

    model = SentenceTransformer('all-mpnet-base-v2')
    distracted_similarities = []

    for i, sentence in enumerate(sentences):
        if any(dw.lower() in sentence.lower() for dw in distractor_words):
            neighbor_sims = []
            # Compute similarity with previous sentence if exists
            if i > 0:
                try:
                    sim_prev = util.cos_sim(
                        model.encode(sentence, convert_to_tensor=True),
                        model.encode(sentences[i-1].strip(), convert_to_tensor=True)
                    ).item()
                    neighbor_sims.append(sim_prev)
                except Exception as e:
                    print("Error computing similarity with previous sentence:", e)
            # Compute similarity with next sentence if exists
            if i < len(sentences) - 1:
                try:
                    sim_next = util.cos_sim(
                        model.encode(sentence, convert_to_tensor=True),
                        model.encode(sentences[i+1].strip(), convert_to_tensor=True)
                    ).item()
                    neighbor_sims.append(sim_next)
                except Exception as e:
                    print("Error computing similarity with next sentence:", e)
            if neighbor_sims:
                distracted_similarities.append(np.mean(neighbor_sims))
    
    if not distracted_similarities:
        return 10.0

    avg_similarity = np.mean(distracted_similarities)
    smoothness_score = avg_similarity * 10
    smoothness_score = max(0, min(smoothness_score, 10))
    return round(smoothness_score, 2)

def analyze_topic_adherence(transcript: str, expected_topic: str) -> float:
    """
    Analyze topic adherence by:
      1. Splitting the transcript into sentences.
      2. Computing embeddings for each non-empty sentence using SentenceTransformer.
      3. Computing the embedding for the expected topic.
      4. Calculating the cosine similarity between each sentence and the expected topic.
      5. Aggregating these similarities into a global topic adherence score (0-10).
      
    Returns a score between 0 and 10.
    """
    sentences = sent_tokenize(transcript)
    if not sentences:
        return 0.0

    model = SentenceTransformer('all-mpnet-base-v2')
    try:
        topic_embedding = model.encode(expected_topic, convert_to_tensor=True)
    except Exception as e:
        print("Error encoding expected_topic:", e)
        return 0.0

    similarities = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        try:
            sentence_embedding = model.encode(sentence, convert_to_tensor=True)
        except Exception as e:
            print("Error encoding sentence:", sentence, e)
            continue
        # Check if embedding is valid (non-empty)
        if sentence_embedding is None or sentence_embedding.shape[0] == 0:
            print("Empty embedding for sentence:", sentence)
            continue
        try:
            cos_sim = util.cos_sim(topic_embedding, sentence_embedding).item()
        except Exception as e:
            print("Error computing cosine similarity for sentence:", sentence, e)
            continue
        similarities.append(cos_sim)

    if not similarities:
        return 0.0

    avg_similarity = np.mean(similarities)
    score = avg_similarity * 10
    score = max(0, min(score, 10))
    return round(score, 2)

def analyze_coherence(transcript):
    """
    Analyze speech coherence by:
      1. Segmenting the transcript into sentences.
      2. Embedding each sentence using Sentence-BERT.
      3. Computing cosine similarity between adjacent sentence embeddings.
      4. Aggregating these similarities into a global coherence score.
      
    Returns:
      - coherence_score: A score (0-10) representing global coherence.
      - sentences: List of segmented sentences.
      - similarities: List of cosine similarities between adjacent sentences.
    """
    # Sentence Segmentation
    sentences = nltk.sent_tokenize(transcript)
    if len(sentences) < 2:
        return 10.0, sentences, []
    
    # Embed Sentences with Sentence-BERT
    model = SentenceTransformer('all-mpnet-base-v2')
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Compute Cosine Similarity Between Adjacent Sentences
    similarities = []
    for i in range(len(sentences) - 1):
        cos_sim = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i+1]).item()
        similarities.append(cos_sim)
    
    # Aggregate Similarities: Compute the average similarity
    avg_similarity = np.mean(similarities)

    coherence_score = avg_similarity * 10
    
    return coherence_score

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



def generate_topics():
    """
    Generate a main speaking topic and contextually relevant distractor words.
    The response from the model is expected to contain lines like:
    
        Main Topic: <topic text>
        Distractor Words: <word1>, <word2>, <word3>
        
    Returns:
        dict: {"main_topic": <str>, "distractor_words": [<str>, ...]}
    """
    client = Groq(api_key="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")
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": ("You are a helpful AI Assistant. You should generate and display a main speaking topic, like Cars tourism travelling  "
                            "and then generate contextually relevant distractor words related to that topic. "
                            "Please output the result in the following format exactly:\n\n"
                            "Main Topic: <your topic here>\n"
                            "Distractor Words: <word1>, <word2>, <word3>")
            },
            {
                "role": "user",
                "content": "Give a main topic and some distractor words."
            }
        ],
        temperature=2,
        frequency_penalty=0.0,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    response_content = ""
    for chunk in completion:
        # Each chunk's delta content may be None; we append if present.
        delta_text = chunk.choices[0].delta.content or ""
        response_content += delta_text
        print(delta_text, end="") 

    main_topic = ""
    distractor_words = []
    lines = response_content.splitlines()
    for line in lines:
        if line.lower().startswith("main topic:"):
            main_topic = line.split(":", 1)[1].strip()
        elif line.lower().startswith("distractor words:"):
            words_str = line.split(":", 1)[1].strip()
            distractor_words = [w.strip() for w in words_str.split(",") if w.strip()]
    
    return {"main_topic": main_topic, "distractor_words": distractor_words}

def process_triple_step_audio(audio_file_path, main_topic, distractor_words):

    transcript = process_audio_upload(audio_file_path)
    
    # evaluation scores
    coherence_result = analyze_coherence(transcript)
    if isinstance(coherence_result, tuple):
        coherence_score = coherence_result[0]
    else:
        coherence_score = coherence_result
        
    topic_adherence_score = analyze_topic_adherence(transcript, main_topic)
    distraction_handling_score = analyze_distractor_smoothness(transcript,distractor_words)
    overall_triple_step_score = (coherence_score + topic_adherence_score + distraction_handling_score) / 3.0
    
    
    result = {
        "main_topic": main_topic,
        "distractor_words": distractor_words,
        "transcript": transcript,
        "coherence_score": coherence_score,
        "topic_adherence_score": topic_adherence_score,
        "distraction_handling_score": distraction_handling_score,
        "overall_triple_step_scrore": overall_triple_step_score
    }
    return result
