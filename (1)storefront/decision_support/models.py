from django.db import models

# Create your models here.
class water_data(models.Model):
    Date = models.DateField()
    WaterLevel = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    Rainfall = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    Drawdown = models.IntegerField(null=True)


class rainfall_data(models.Model):
    Date = models.DateField()
    Rainfall = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    MaxTemp = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    MinTemp = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    MeanTemp = models.DecimalField(decimal_places=2, max_digits=5, null=True)
    WindSpeed = models.IntegerField(null=True)
    WindDirection = models.IntegerField(null=True)   
    RelativeHumidity = models.IntegerField(null=True)   


class business_zones(models.Model):
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
    