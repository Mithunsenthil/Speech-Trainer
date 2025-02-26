from collections import Counter
from transformers import pipeline
from groq import Groq
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import numpy as np
from .audio_process import process_audio_upload, convert_wav_to_mp3, transcribe_audio, save_uploaded_file 

def generate_incomplete_analogy():
    client = Groq(api_key="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")
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI Assistant. "
                    "Generate a single incomplete analogy prompt. "
                    "For example: 'Learning is like', 'Love is like'. "
                    "Output exactly one incomplete analogy without any additional words."
                ) 
            },
            {
            "role": "user",
            "content": "give a random incomplete anology",
            }
        ],
        temperature=2,
        frequency_penalty=0.0,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )   
    print(completion)
    # Handling the streamed output
    response_content = ""
    for chunk in completion:
        response_content = response_content+(chunk.choices[0].delta.content or "")
        print(chunk.choices[0].delta.content or "", end="")

    return response_content

def extract_topic(transcript):

    # Initialize english stopwords
    english_stopwords = stopwords.words("english")

    #convert article to tokens
    tokens = word_tokenize(transcript)

    #extract alpha words and convert to lowercase
    alpha_lower_tokens = [word.lower() for word in tokens if word.isalpha()]

    #remove stopwords
    alpha_no_stopwords = [word for word in alpha_lower_tokens if word not in english_stopwords]

    #Count word
    BoW = Counter(alpha_no_stopwords)

    #Most common words
    return BoW.most_common(3) 

def score_analogy_with_groq(transcript, incomplete_analogy):
    """
    Use Groq to evaluate analogy relevance and creativity based on the transcript
    and the incomplete analogy prompt.
    
    Parameters:
        transcript (str): The transcribed text from the user's response
        incomplete_analogy (str): The incomplete analogy prompt (e.g., "Success is like")
        
    Returns:
        dict: A dictionary with analogy_relevance and creativity scores
    """
    client = Groq(api_key="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")
    
    prompt = f"""
    Below is an incomplete analogy prompt and a user's spoken response to complete it.
    
    Incomplete Analogy: "{incomplete_analogy}"
    
    User's Response: "{transcript}"
    
    Please evaluate the response based on two criteria:
    
    1. Relevance (0-10): How well does the response connect to the analogy prompt? 
       Does it create a clear and appropriate comparison?
       
    2. Creativity (0-10): How original, insightful, or thought-provoking is the analogy?
       Does it provide a fresh perspective or use unexpected connections?
    
    Return your evaluation as a JSON object with two properties:
    - analogy_relevance: a number between 0 and 10 (with up to 2 decimal places)
    - creativity: a number between 0 and 10 (with up to 2 decimal places)
    
    Response must be in this exact JSON format and nothing else:
    {{
        "analogy_relevance": 0.00,
        "creativity": 0.00
    }}
    """
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that evaluates analogies based on relevance and creativity. Provide numeric scores only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.8,  
        max_completion_tokens=256
    )
    
    response_content = completion.choices[0].message.content
    response_dict = json.loads(response_content)

    return {
        "analogy_relevance": round(response_dict['analogy_relevance'], 2),
        "creativity": round(response_dict["creativity"], 2)
    }
        
def process_rapidfire_audio(audio_file_path, analogy):

    result = process_audio_upload(audio_file_path)
    segments = result.get("segments", [])
    transcript = result.get("text", "")
    
    # Calculate word-level timestamps
    word_timestamps = []
    previous_word_end = None
    gaps = []  # store gaps between consecutive words
    
    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_text = segment["text"].strip()
        words = seg_text.split()
        num_words = len(words)
        if num_words == 0:
            continue
        duration = seg_end - seg_start
        word_duration = duration / num_words
        
        for i, word in enumerate(words):
            word_start = seg_start + i * word_duration
            word_end = word_start + word_duration
            word_timestamps.append({
                "word": word,
                "start": round(word_start, 2),
                "end": round(word_end, 2)
            })
            if previous_word_end is not None:
                gap = word_start - previous_word_end
                gaps.append(gap)
            previous_word_end = word_end

    if transcript.strip() == "":
        speech_continuity = 0
    else:
        if gaps:
            gaps_array = np.array(gaps)
            avg_gap = np.mean(gaps_array)
            std_gap = np.std(gaps_array)
            
            # Penalize both large average gaps and inconsistent timing
            speech_continuity = max(0, 10 - (avg_gap * 5) - (std_gap * 3))
        else:
            speech_continuity = 10.0

    # Score analogy relevance and creativity using Groq
    analogy_scores = score_analogy_with_groq(transcript, analogy)
    analogy_relevance = analogy_scores["analogy_relevance"]
    creativity = analogy_scores["creativity"]
    
    # Extract text topic using NLP
    text_topic = extract_topic(transcript)
    
    # Calculate overall score
    overall_rapidfire_score = (speech_continuity + analogy_relevance + creativity) / 3
    
    metrics = {
        "speech_continuity": round(speech_continuity, 2),
        "analogy_relevance": analogy_relevance,
        "creativity": creativity,
        "overall_rapidfire_score": round(overall_rapidfire_score, 2),
        "text_topic": text_topic
    }
    return {
        "transcript": transcript,
        "word_timestamps": word_timestamps,
        "metrics": metrics,
        "generated_analogy": analogy
    }
