# exercises/views.py

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.core.files.storage import default_storage
from django.contrib.auth.decorators import login_required
from datetime import timedelta
from django.utils.timezone import now
from django.db.models import Avg, Sum
import os
import tempfile

# Import your utility functions
from .Utils.rapidfire import process_rapidfire_audio, generate_incomplete_analogy
from .Utils.triplestep import process_triple_step_audio, generate_topics
from .Utils.conductor import process_conductor_audio, generate_conductor_exercise
from .Utils.xp_system import calculate_user_xp, calculate_user_level

# Import your models (ensure these models are defined in your models.py)
from .models import RapidFire, TripleStep, Conductor, UserProfile

# class dashboard(TemplateView):
#     template_name = "dashboard.html"

class dashboard(TemplateView):
    template_name = "dashboard.html"
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user     
        if user.is_authenticated:
            total_xp = calculate_user_xp(user)
            level = calculate_user_level(total_xp)
            
        context.update({
            "total_xp": total_xp,
            "level": level,
        })
        return context
    

@login_required
def get_radar_chart_data(request):
    user = request.user
    # For RapidFire: average the sub-metrics.
    rapidfire = RapidFire.objects.filter(user=user).aggregate(
        speech_continuity=Avg('speech_continuity'),
        analogy_relevance=Avg('analogy_relevance'),
        creativity=Avg('creativity')
    )
    # For TripleStep:
    triplestep = TripleStep.objects.filter(user=user).aggregate(
        coherence_score=Avg('coherence_score'),
        topic_adherence_score=Avg('topic_adherence_score'),
        distraction_handling_score=Avg('distraction_handling_score')
    )
    # For Conductor:
    conductor = Conductor.objects.filter(user=user).aggregate(
        energy_level_score=Avg('energy_level_score'),
        mood_match_score=Avg('mood_match_score')
    )
    data = {
        "rapidfire": rapidfire,
        "triplestep": triplestep,
        "conductor": conductor,
    }
    return JsonResponse(data)


@login_required
def get_pie_chart_data(request):
    """ Fetch XP distribution across exercises """
    user = request.user
    rapidfire_xp = RapidFire.objects.filter(user=user).aggregate(total=Sum('overall_rapidfire_score'))['total'] or 0
    triplestep_xp = TripleStep.objects.filter(user=user).aggregate(total=Sum('overall_triple_step_scrore'))['total'] or 0
    conductor_xp = Conductor.objects.filter(user=user).aggregate(total=Sum('overall_conductor_score'))['total'] or 0

    pie_data = [
        {"exercise": "RapidFire", "xp": rapidfire_xp},
        {"exercise": "TripleStep", "xp": triplestep_xp},
        {"exercise": "Conductor", "xp": conductor_xp},
    ]
    return JsonResponse(pie_data, safe=False)


@login_required
def get_bar_chart_data(request):
    """ Fetch average scores per exercise """
    user = request.user
    rapidfire_avg = RapidFire.objects.filter(user=user).aggregate(avg=Avg('overall_rapidfire_score'))['avg'] or 0
    triplestep_avg = TripleStep.objects.filter(user=user).aggregate(avg=Avg('overall_triple_step_scrore'))['avg'] or 0
    conductor_avg = Conductor.objects.filter(user=user).aggregate(avg=Avg('overall_conductor_score'))['avg'] or 0

    bar_data = [
        {"exercise": "RapidFire", "avg_score": rapidfire_avg},
        {"exercise": "TripleStep", "avg_score": triplestep_avg},
        {"exercise": "Conductor", "avg_score": conductor_avg},
    ]
    return JsonResponse(bar_data, safe=False)

@login_required
def line_chart_data(request):
    user = request.user

    # Fetch user scores for all tests
    rapidfire_scores = list(RapidFire.objects.filter(user=user).values_list('overall_rapidfire_score', flat=True))
    triplestep_scores = list(TripleStep.objects.filter(user=user).values_list('overall_triple_step_scrore', flat=True))
    conductor_scores = list(Conductor.objects.filter(user=user).values_list('overall_conductor_score', flat=True))

    # Generate test numbers
    test_numbers = list(range(1, max(len(rapidfire_scores), len(triplestep_scores), len(conductor_scores)) + 1))

    return JsonResponse({
        "test_numbers": test_numbers,
        "rapidfire_scores": rapidfire_scores,
        "triplestep_scores": triplestep_scores,
        "conductor_scores": conductor_scores
    })

def leaderboard_data(request):
    """
    Fetch the top 5 users with the highest XP.
    """
    top_users = UserProfile.objects.order_by('-total_xp')[:5]
    data = [
        {
            'username': user.user.username,
            'xp': user.total_xp
        }
        for user in top_users
    ]
    return JsonResponse(data, safe=False)


# --- Rapid Fire Views ---
def rapidfire(request):
    analogy = generate_incomplete_analogy()
    print("from views", analogy)
    
    level = None
    total_xp = None
    if request.user.is_authenticated:
        total_xp = calculate_user_xp(request.user)
        level = calculate_user_level(total_xp)
    
    return render(request, "rapidfire.html", {
        "rapidfire_analogy": analogy,
        "level": level,
        "total_xp": total_xp
    })

@csrf_exempt
def rapidfire_submit(request):
    if request.method == "POST" and "audio" in request.FILES:
        # Retrieve the analogy from the POST data
        analogy = request.POST.get("analogy", "")

        audio_file = request.FILES["audio"]
        temp_path = os.path.join(tempfile.gettempdir(), audio_file.name)
        with open(temp_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
                
        # Process the rapidfire audio to get transcript, word timestamps, and metrics
        result = process_rapidfire_audio(temp_path, analogy)
        
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Save the result in the session
        request.session["rapidfire_result"] = result
        
        # Store the computed scores into the RapidFire model.
        # We cast (or round) the float scores to integer as the model uses IntegerFields.
        RapidFire.objects.create(
            user=request.user,
            speech_continuity=int(round(result["metrics"].get("speech_continuity", 0))),
            analogy_relevance=int(round(result["metrics"].get("analogy_relevance", 0))),
            creativity=int(round(result["metrics"].get("creativity", 0))),
            overall_rapidfire_score=int(round(result["metrics"].get("overall_rapidfire_score", 0)))
        )
        
        return redirect("rapidfire_result")

    return JsonResponse({"error": "Invalid request"}, status=400)

def rapidfire_result(request):
    result = request.session.get("rapidfire_result", {})
    level = None
    total_xp = None
    if request.user.is_authenticated:
        total_xp = calculate_user_xp(request.user)
        level = calculate_user_level(total_xp)
    print("Rapidfire result from session:", result)
    return render(request, "rapidfire_result.html", {"rapidfire_result": result, "level": level, "total_xp": total_xp})

# --- Triple Step Views ---
def triple_step(request):
    """
    Render the Triple Step exercise page.
    A main speaking topic and an initial set of distractor words are chosen.
    """
    result = generate_topics()
    level = None
    total_xp = None
    if request.user.is_authenticated:
        total_xp = calculate_user_xp(request.user)
        level = calculate_user_level(total_xp)
    
    return render(request, "triplestep.html", {
        "main_topic": result["main_topic"],
        "distractor_words": result["distractor_words"],
        "level": level,
        "total_xp": total_xp
    })

@csrf_exempt
def triple_step_submit(request):
    """
    Handle the POST request from the Triple Step exercise.
    The audio file is saved temporarily and processed using our triple step logic.
    """
    if request.method == "POST" and "audio" in request.FILES:
        main_topic = request.POST.get("main_topic", "")
        distractor_words = request.POST.get("distractor_words", "")

        audio_file = request.FILES["audio"]
        temp_path = os.path.join(tempfile.gettempdir(), audio_file.name)
        with open(temp_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        
        result = process_triple_step_audio(temp_path, main_topic, distractor_words)

        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        request.session["triplestep_result"] = result
        
        # Save the computed scores into the TripleStep model.
        TripleStep.objects.create(
            user=request.user,
            coherence_score=int(round(result.get("coherence_score", 0))),
            topic_adherence_score=int(round(result.get("topic_adherence_score", 0))),
            distraction_handling_score=int(round(result.get("distraction_handling_score", 0))),
            overall_triple_step_scrore=int(round(result.get("overall_triple_step_scrore", 0)))
        )
        
        return redirect("triplestep_result")
    
    return JsonResponse({"error": "Invalid request"}, status=400)

def triple_step_result(request):
    result = request.session.get("triplestep_result", {})
    level = None
    total_xp = None
    if request.user.is_authenticated:
        total_xp = calculate_user_xp(request.user)
        level = calculate_user_level(total_xp)
    print(result)
    return render(request, "triplestep_result.html", {"result": result, "level": level, "total_xp": total_xp})

# --- Conductor Views ---
def conductor(request):
    """
    Render the Conductor exercise page.
    This page guides the user through different energy levels and moods.
    """
    context = generate_conductor_exercise()
    result = generate_topics()
    level = None
    total_xp = None
    if request.user.is_authenticated:
        total_xp = calculate_user_xp(request.user)
        level = calculate_user_level(total_xp)
    return render(request, "conductor.html", {
        "energy_levels": context["energy_levels"],
        "moods": context["moods"],
        "improvement_suggestions": context["improvement_suggestions"],
        "level": level,
        "total_xp": total_xp,
    })
        
@csrf_exempt
def conductor_submit(request):
    """
    Handle the Conductor exercise POST submission.
    The audio file is saved temporarily and processed to simulate real-time voice analysis.
    """
    if request.method == "POST" and "audio" in request.FILES:
        instructions = request.POST.get("improvement_suggestions", "")
        energy_levels = request.POST.get("energy_levels", "")
        moods = request.POST.get("moods", "")

        audio_file = request.FILES["audio"]
        temp_path = os.path.join(tempfile.gettempdir(), audio_file.name)
        with open(temp_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        
        result = process_conductor_audio(temp_path, instructions, energy_levels, moods)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        request.session["conductor_result"] = result
        
        # Save the computed scores into the Conductor model.
        Conductor.objects.create(
            user=request.user,
            energy_level_score=int(round(result.get("energy_level_score", 0))),
            mood_match_score=int(round(result.get("mood_match_score", 0))),
            overall_conductor_score=int(round(result.get("overall_conductor_score", 0)))
        )
        
        return redirect("conductor_result")
    
    return JsonResponse({"error": "Invalid request"}, status=400)

def conductor_result(request):
    result = request.session.get("conductor_result", {})
    level = None
    total_xp = None
    if request.user.is_authenticated:
        total_xp = calculate_user_xp(request.user)
        level = calculate_user_level(total_xp)
    print(result)
    print(result)
    return render(request, "conductor_result.html", {"result": result, "level": level, "total_xp": total_xp})
