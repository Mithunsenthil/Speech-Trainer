from django.urls import path
from .views import *

urlpatterns = [
    # path('rapidfire', views.rapidfire, name='rapidfire'),
    path('dashboard/', dashboard.as_view(), name='dashboard'),
    path('rapidfire/submit/', rapidfire_submit, name='rapidfire_submit'),
    path('rapidfire/result/', rapidfire_result, name='rapidfire_result'),
    path('rapidfire/', rapidfire, name='rapidfire'),
    path("triple_step/", triple_step, name="triplestep"),
    path("triple_step/submit/", triple_step_submit, name="triplestep_submit"),
    path("triple_step/result/", triple_step_result, name="triplestep_result"),
    path("conductor/", conductor, name="conductor"),
    path("conductor/submit/", conductor_submit, name="conductor_submit"),
    path("conductor/result/", conductor_result, name="conductor_result"),
    # path('dashboard_data/', dashboard_data, name='dashboard_data'),
    path("api/pie-chart-data/", get_pie_chart_data, name="pie-chart-data"),
    path("api/bar-chart-data/", get_bar_chart_data, name="bar-chart-data"),
    path('api/leaderboard-data/', leaderboard_data, name='leaderboard_data'),
    path('api/line-chart-data/', line_chart_data, name='line_chart_data'),
    path('api/radar-chart-data/', get_radar_chart_data, name='radar-chart-data'),


]