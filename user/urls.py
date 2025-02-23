from django.urls import path
from .views import register, editprofile 
from . import views

urlpatterns = [
    path('register/',register.as_view(),name='register'),
    path('edit_profile/',editprofile.as_view(),name='edit_profile'),
    path('login/', views.login_view, name='login'),
    
]

