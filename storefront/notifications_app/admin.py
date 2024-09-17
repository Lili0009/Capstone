from django.contrib import admin
from django_celery_beat.models import SolarSchedule, ClockedSchedule, IntervalSchedule

# Register your models here.
from .models import BroadcastNotification
admin.site.register(BroadcastNotification)

admin.site.unregister(SolarSchedule)
admin.site.unregister(ClockedSchedule)
admin.site.unregister(IntervalSchedule)