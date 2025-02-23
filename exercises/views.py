from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from .Utils.rapidfire import process_audio_upload,analyze_transcript
from .Utils.triplestep import process_audio_upload, process_triple_step_audio
from .Utils.conductor import process_audio_upload, process_conductor_audio
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
import os
import tempfile

class dashboard(TemplateView):
    template_name = "dashboard.html"

#Rapid Fire
class rapidfire(TemplateView):
    template_name = "rapidfire.html"

@csrf_exempt
def rapidfire_submit(request):
    if request.method == "POST" and "audio" in request.FILES:
        audio_file = request.FILES["audio"]

        # Create a temporary file path in the system's temp directory
        temp_path = os.path.join(tempfile.gettempdir(), audio_file.name)

        # Write file directly using built-in open(), bypassing default_storage
        with open(temp_path, "wb") as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)

        # Process the file using your custom processing function
        response_data = process_audio_upload(temp_path)
        analysis_data = analyze_transcript(response_data)

        # Delete the temporary file after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Save result in session and redirect to the result page
        request.session["rapidfire_submit"] = [response_data,analysis_data]
        return redirect("rapidfire_result")

    return JsonResponse({"error": "Invalid request"}, status=400)

def rapidfire_result(request):
    # Get the stored result from session
    result = request.session.get("rapidfire_submit", {})

    return render(request, "rapidfire_result.html", {"result": result})


# Triple Step

def triple_step(request):
    """
    Render the Triple Step exercise page.
    A main speaking topic and an initial set of distractor words are chosen.
    These are stored in the session for later use.
    """
    main_topic = "Discuss the impact of technology on modern education."
    distractor_words = ["distraction", "interrupt", "side-track", "noise"]
    request.session["main_topic"] = main_topic
    request.session["distractor_words"] = distractor_words
    return render(request, "triplestep.html", {
        "main_topic": main_topic,
        "distractor_words": distractor_words
    })

@csrf_exempt
def triple_step_submit(request):
    """
    Handle the POST request from the Triple Step exercise.
    The audio file is saved temporarily and processed using our triple step logic.
    """
    if request.method == "POST" and "audio" in request.FILES:
        audio_file = request.FILES["audio"]
        temp_path = os.path.join(tempfile.gettempdir(), audio_file.name)
        with open(temp_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        
        # Retrieve topic and distractors from session
        main_topic = request.session.get("main_topic", "Default Topic")
        distractor_words = request.session.get("distractor_words", [])
        
        # Process the audio file with Triple Step logic:
        result = process_triple_step_audio(temp_path, main_topic, distractor_words)
        
        # Remove the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Save result in session and redirect to result page
        request.session["triplestep_result"] = result
        return redirect("triplestep_result")
    
    return JsonResponse({"error": "Invalid request"}, status=400)

def triple_step_result(request):
    """
    Render the results page for the Triple Step exercise.
    """
    result = request.session.get("triplestep_result", {})
    return render(request, "triplestep_result.html", {"result": result})



# --- Conductor Exercise Views ---

def conductor(request):
    """
    Render the Conductor exercise page.
    This page guides the user through different energy levels and moods.
    """
    # Define a set of energy levels and moods (could be randomized or fixed)
    energy_levels = ["Low", "Moderate", "High"]
    moods = ["Calm", "Happy", "Excited", "Serious"]
    context = {
        "energy_levels": energy_levels,
        "moods": moods,
        "instructions": (
            "For this exercise, please record your voice as you adjust your energy and mood. "
            "Try to match the energy level and mood shown on the screen. Your performance will be analyzed."
        )
    }
    return render(request, "conductor.html", context)

@csrf_exempt
def conductor_submit(request):
    """
    Handle the Conductor exercise POST submission.
    The audio file is saved temporarily and processed to simulate real-time voice analysis.
    """
    if request.method == "POST" and "audio" in request.FILES:
        audio_file = request.FILES["audio"]
        temp_path = os.path.join(tempfile.gettempdir(), audio_file.name)
        with open(temp_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        
        # Process the audio file with Conductor logic (simulated)
        result = process_conductor_audio(temp_path)
        
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Save the result in the session and redirect to the result page
        request.session["conductor_result"] = result
        return redirect("conductor_result")
    
    return JsonResponse({"error": "Invalid request"}, status=400)

def conductor_result(request):
    """
    Render the results page for the Conductor exercise.
    """
    result = request.session.get("conductor_result", {})
    return render(request, "conductor_result.html", {"result": result})
