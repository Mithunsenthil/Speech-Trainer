from django.urls import path
from .views import *

urlpatterns = [
    # path('rapidfire', views.rapidfire, name='rapidfire'),
    path('dashboard/', dashboard.as_view(), name='dashboard'),
    path('rapidfire/submit/', rapidfire_submit, name='rapidfire_submit'),
    path('rapidfire/result/', rapidfire_result, name='rapidfire_result'),
    path('rapidfire/', rapidfire.as_view(), name='rapidfire'),
    path("triple_step/", triple_step, name="triplestep"),
    path("triple_step/submit/", triple_step_submit, name="triplestep_submit"),
    path("triple_step/result/", triple_step_result, name="triplestep_result"),
    path("conductor/", conductor, name="conductor"),
    path("conductor/submit/", conductor_submit, name="conductor_submit"),
    path("conductor/result/", conductor_result, name="conductor_result"),

]