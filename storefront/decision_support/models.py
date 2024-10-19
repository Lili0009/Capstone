from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from notifications_app.models import BroadcastNotification
from notifications_app.tasks import broadcast_notification
from django.utils import timezone
from datetime import timedelta


class water_data(models.Model):
    Date = models.DateField()
    WaterLevel = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    Rainfall = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    Drawdown = models.IntegerField(null=True)
    

@receiver(post_save, sender=water_data)
def check_water_level(sender, instance, created, **kwargs):
    if created:
        tolerance = 0.01  
        if instance.WaterLevel >= 80.15 - tolerance:
            message = f"Red Alert: Water level reached spilling level {instance.WaterLevel}m."
            notification = BroadcastNotification.objects.create(
                message=message,
                broadcast_on=timezone.now()
            )

            broadcast_notification.delay(notification.id)

        elif instance.WaterLevel <= 69:
            message = f"Red Alert: Critical water level at {instance.WaterLevel}m"
            notification = BroadcastNotification.objects.create(
                message=message,
                broadcast_on=timezone.now()  
            )

            broadcast_notification.delay(notification.id)


class rainfall_data(models.Model):
    Date = models.DateField()
    Rainfall = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    MaxTemp = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    MinTemp = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    MeanTemp = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    WindSpeed = models.IntegerField(null=True)
    WindDirection = models.IntegerField(null=True)   
    RelativeHumidity = models.IntegerField(null=True)   


class business_zones_data(models.Model):
    CHOICES = [
        ('Araneta-Libis', 'Araneta-Libis'),
        ('Elliptical', 'Elliptical'),
        ('San Juan', 'San Juan'),
        ('Tandang sora', 'Tandang sora'),
        ('Timog', 'Timog'),
        ('Up-Katipunan', 'Up-Katipunan'),
    ]
    
    Date = models.DateField()
    Business_zones = models.CharField(max_length=20, choices=CHOICES)
    Supply_volume = models.DecimalField(decimal_places=2, max_digits=5)
    Bill_volume = models.DecimalField(decimal_places=2, max_digits=5)
    