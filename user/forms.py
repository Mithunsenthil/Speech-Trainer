from typing import Any
from django import forms
from django.contrib.auth.forms import UserCreationForm,UserChangeForm
from django.contrib.auth.models import User

class register(UserCreationForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(max_length=225, widget=forms.TimeInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=225, widget=forms.TimeInput(attrs={'class': 'form-control'}))

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name')

    def __init__(self, *args, **kwargs):
        super(register, self).__init__(*args, **kwargs)

        self.fields["username"].widget.attrs['class'] = 'form-control'
        self.fields["password1"].widget.attrs['class'] = 'form-control'
        self.fields["password2"].widget.attrs['class'] = 'form-control'

class editprofile(UserChangeForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(max_length=225, widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=225, widget=forms.TextInput(attrs={'class': 'form-control'}))
    username = forms.CharField(max_length=225, widget=forms.TextInput(attrs={'class': 'form-control'}))

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name' , 'email')