# exercises/utils/xp_system.py
import math
from django.db.models import Sum
from exercises.models import RapidFire, TripleStep, Conductor

def calculate_user_xp(user):

    rapidfire_xp = RapidFire.objects.filter(user=user).aggregate(total=Sum('overall_rapidfire_score'))['total'] or 0
    triplestep_xp = TripleStep.objects.filter(user=user).aggregate(total=Sum('overall_triple_step_scrore'))['total'] or 0
    conductor_xp = Conductor.objects.filter(user=user).aggregate(total=Sum('overall_conductor_score'))['total'] or 0
    
    total_xp = rapidfire_xp + triplestep_xp + conductor_xp
    return total_xp

def calculate_user_level(xp, xp_per_level=50):

    level = math.floor(xp / xp_per_level) + 1
    return level
