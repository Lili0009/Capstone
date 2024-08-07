from django.urls import path
from . import views
from django.contrib import admin
admin.site.site_header = 'Hydrocision Administration'
admin.site.site_title = 'Hydrocision'
admin.site.index_title = 'Hydrocision Site'

urlpatterns = [
    path('', views.Dashboard, name='dashboard'),
    path('forecast/', views.Forecast, name='forecast'),
    path('test/', views.test, name="home"),
    path('business_zone/', views.Business_zone, name='business_zone'),
    path('img_map/', views.Img_map, name='img_map')
]