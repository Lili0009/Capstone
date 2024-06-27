from django.urls import path
from . import views
from django.contrib import admin
admin.site.site_header = 'Hydrocision Administration'
admin.site.site_title = 'Hydrocision'
admin.site.index_title = 'Hydrocision Site'

urlpatterns = [
    path('', views.Dashboard, name='dashboard'),
    path('forecast/', views.Forecast, name='forecast'),
    path('business_zone/', views.Business_zone, name='business_zone')
]