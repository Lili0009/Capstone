from django.contrib import admin
# 👇 1. Add this line import notification model
from .models import Notification

admin.site.register(Notification)