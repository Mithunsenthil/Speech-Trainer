from django.db.models.base import Model as Model
from django.db.models.query import QuerySet
from django.shortcuts import render
from django.views import generic 
from .forms import register, editprofile
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.urls import reverse_lazy
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
# Create your views here.

class register(generic.CreateView):
    form_class = register
    template_name = 'registration/register.html'
    success_url = reverse_lazy('login')

class editprofile(generic.UpdateView):
    form_class = editprofile
    template_name = 'registration/editprofile.html'
    success_url = reverse_lazy('dashboard')

    def get_object(self):
        return self.request.user
    
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')  # Redirect to dashboard after login
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})