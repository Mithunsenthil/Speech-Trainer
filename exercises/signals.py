from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import RapidFire, TripleStep, Conductor, UserProfile

@receiver(post_save, sender=RapidFire)
@receiver(post_save, sender=TripleStep)
@receiver(post_save, sender=Conductor)
def update_user_xp(sender, instance, **kwargs):
    user_profile, created = UserProfile.objects.get_or_create(user=instance.user)
    user_profile.update_xp()
