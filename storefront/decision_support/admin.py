from django.contrib import admin
from . models import *
import csv
from django.urls import path
from django.shortcuts import render
from django import forms
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.urls import reverse
from datetime import datetime

class CsvImportForm(forms.Form):
    csv_upload = forms.FileField()



class water_data_admin(admin.ModelAdmin):
    list_display = ('Date', 'WaterLevel', 'Rainfall', 'Drawdown')

    def get_urls(self):
        urls = super().get_urls()
        new_urls = [path('upload-csv/', self.upload_csv),]
        return new_urls + urls

    def upload_csv(self, request):
        if request.method == "POST":
            csv_file = request.FILES.get("csv_upload")

            if not csv_file:
                messages.error(request, "No file selected.")
                return HttpResponseRedirect(request.path_info)
            
            if not csv_file.name.endswith('.csv'):
                messages.warning(request, 'The wrong file type was uploaded')
                return HttpResponseRedirect(request.path_info)
            
            try:
                file_data = csv_file.read().decode("utf-8")
                csv_data = csv.reader(file_data.split("\n"))
                header_skipped = False
                for row in csv_data:
                    if not header_skipped:
                        header_skipped = True
                        continue 

                    if len(row) == 0 or row[0].strip() == "":  
                        continue
                    if len(row) != 4:
                        messages.error(request, f"Malformed row: {','.join(row)}")
                        continue  



                    date_str = row[0]
                    try:
                        date_obj = datetime.strptime(date_str, '%d-%b-%y').date()
                    except ValueError:
                        messages.error(request, f"Date format error in row: {','.join(row)}")
                        continue

                    #To accept null values when importing csv file to the administration site
                    water_level = row[1].strip() if row[1].strip() else None
                    rainfall = row[2].strip() if row[2].strip() else None
                    drawdown = row[3].strip() if row[3].strip() else None


                    try:
                        water_level = float(water_level) if water_level else None
                        rainfall = float(rainfall) if rainfall else None
                        drawdown = float(drawdown) if drawdown else None
                    except ValueError:
                        messages.error(request, f"Invalid numeric value in row: {','.join(row)}")
                        continue

                    #To add the objects in administration site
                    water_data.objects.update_or_create(
                        Date=date_obj,
                        defaults={
                            'WaterLevel': water_level,
                            'Rainfall': rainfall,
                            'Drawdown': drawdown,
                        }
                    )
                messages.success(request, "CSV file has been processed successfully.")
            except Exception as e:
                messages.error(request, f"Error processing file: {e}")
            return HttpResponseRedirect(request.path_info)

        form = CsvImportForm()
        data = {"form": form}
        
        return render(request, "admin/csv_upload.html", data)

    def add_new_data(self, request, queryset):
        with open('water_data.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for i, obj in enumerate(queryset):
                formatted_date = obj.Date.strftime('%d-%b-%y')
                if i > 0:
                    writer.writerow([])
                writer.writerow([formatted_date, obj.WaterLevel, obj.Rainfall, obj.Drawdown])  
        self.message_user(request, "Records added to CSV file successfully.")
    add_new_data.short_description = "Add selected records to CSV file"

    actions = ['add_new_data']
admin.site.register(water_data, water_data_admin)



class rainfall_data_admin(admin.ModelAdmin):
    list_display = ('Date', 'Rainfall', 'MaxTemp', 'MinTemp', 'MeanTemp', 'WindSpeed', 'WindDirection', 'RelativeHumidity')
    def add_new_data(self, request, queryset):
        with open('rainfall_data.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for i, obj in enumerate(queryset):
                formatted_date = obj.Date.strftime('%d-%b-%y')
                date_object = datetime.strptime(formatted_date, '%d-%b-%y')
                year = date_object.year
                month = date_object.month
                day = date_object.day
                if i > 0:
                    writer.writerow([])
                writer.writerow([year, month, day, obj.Rainfall, obj.MaxTemp, obj.MinTemp, obj.MeanTemp, obj.WindSpeed, obj.WindDirection, obj.RelativeHumidity])  
        self.message_user(request, "Records added to CSV file successfully.")
    add_new_data.short_description = "Add selected records to CSV file"

    actions = ['add_new_data']
admin.site.register(rainfall_data, rainfall_data_admin)





class business_zones_admin(admin.ModelAdmin):
    list_display = ('Date', 'Business_zones', 'Supply_volume', 'Bill_volume')
    def add_new_data(self, request, queryset):
        with open('manila_water_data.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for i, obj in enumerate(queryset):
                formatted_date = obj.Date.strftime('%d-%b-%y')
                if i > 0:
                    writer.writerow([])
                writer.writerow([formatted_date, obj.Business_zones, obj.Supply_volume, obj.Bill_volume])  
        self.message_user(request, "Records added to CSV file successfully.")
    add_new_data.short_description = "Add selected records to CSV file"

    actions = ['add_new_data']
admin.site.register(business_zones, business_zones_admin)
