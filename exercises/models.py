from django.db import models
from django.contrib.auth.models import User
from django.db.models import Sum, Avg
from django.urls import reverse
# Create your models here.

class RapidFire(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    speech_continuity = models.IntegerField()
    analogy_relevance = models.IntegerField()
    creativity = models.IntegerField()
    overall_rapidfire_score = models.IntegerField()

class TripleStep(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    coherence_score = models.IntegerField()
    topic_adherence_score = models.IntegerField()
    distraction_handling_score = models.IntegerField()
    overall_triple_step_scrore = models.IntegerField()

class Conductor(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    energy_level_score = models.IntegerField()
    mood_match_score = models.IntegerField()
    overall_conductor_score = models.IntegerField()

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    total_xp = models.IntegerField(default=0)

    def update_xp(self):
        """
        Update the total XP for the user by recalculating it.
        """
        self.total_xp = (
            RapidFire.objects.filter(user=self.user).aggregate(Sum('overall_rapidfire_score'))['overall_rapidfire_score__sum'] or 0
        ) + (
            TripleStep.objects.filter(user=self.user).aggregate(Sum('overall_triple_step_scrore'))['overall_triple_step_scrore__sum'] or 0
        ) + (
            Conductor.objects.filter(user=self.user).aggregate(Sum('overall_conductor_score'))['overall_conductor_score__sum'] or 0
        )
        self.save()

