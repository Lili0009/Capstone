from django.contrib import admin
from .models import water_data, rainfall_data, business_zones_data
import csv
from django.urls import path
from django.shortcuts import render
from django import forms
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.urls import reverse
from datetime import datetime, date

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

    def get_urls(self):
        urls = super().get_urls()
        new_urls = [path('upload-csv/', self.upload_csv, name='rainfall_data_upload_csv'),]
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
                    if len(row) != 10:  
                        messages.error(request, f"Malformed row: {','.join(row)}")
                        continue

                    year = int(row[0])
                    month = int(row[1])
                    day = int(row[2])
                    try:
                        date_obj = date(year, month, day)
                    except ValueError:
                        messages.error(request, f"Date format error in row: {','.join(row)}")
                        continue

                    # To accept null values when importing csv file to the administration site
                    rainfall = row[3].strip() if row[1].strip() else None
                    max_temp = row[4].strip() if row[2].strip() else None
                    min_temp = row[5].strip() if row[3].strip() else None
                    mean_temp = row[6].strip() if row[4].strip() else None
                    wind_speed = row[7].strip() if row[5].strip() else None
                    wind_direction = row[8].strip() if row[6].strip() else None
                    relative_humidity = row[9].strip() if row[7].strip() else None

                    try:
                        rainfall = float(rainfall) if rainfall else None
                        max_temp = float(max_temp) if max_temp else None
                        min_temp = float(min_temp) if min_temp else None
                        mean_temp = float(mean_temp) if mean_temp else None
                        wind_speed = float(wind_speed) if wind_speed else None
                        wind_direction = float(wind_direction) if wind_direction else None
                        relative_humidity = float(relative_humidity) if relative_humidity else None
                    except ValueError:
                        messages.error(request, f"Invalid numeric value in row: {','.join(row)}")
                        continue

                    # To add the objects in administration site
                    rainfall_data.objects.update_or_create(
                        Date=date_obj,
                        defaults={
                            'Rainfall': rainfall,
                            'MaxTemp': max_temp,
                            'MinTemp': min_temp,
                            'MeanTemp': mean_temp,
                            'WindSpeed': wind_speed,
                            'WindDirection': wind_direction,
                            'RelativeHumidity': relative_humidity,
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

    def get_urls(self):
        urls = super().get_urls()
        new_urls = [path('upload-csv/', self.upload_csv, name='business_zones_upload_csv'),]
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
                    if len(row) != 4:  # Adjust this if you have a different number of columns
                        messages.error(request, f"Malformed row: {','.join(row)}")
                        continue

                    date_str = row[0]
                    try:
                        date_obj = datetime.strptime(date_str, '%d-%b-%y').date()
                    except ValueError:
                        messages.error(request, f"Date format error in row: {','.join(row)}")
                        continue

                    # To accept null values when importing csv file to the administration site
                    business_zones_value = row[1].strip() if row[1].strip() else None
                    supply_volume = row[2].strip() if row[2].strip() else None
                    bill_volume = row[3].strip() if row[3].strip() else None

                    try:
                        supply_volume = float(supply_volume) if supply_volume else None
                        bill_volume = float(bill_volume) if bill_volume else None
                    except ValueError:
                        messages.error(request, f"Invalid numeric value in row: {','.join(row)}")
                        continue

                    # To add the objects in administration site
                    business_zones_data.objects.update_or_create(
                        Date=date_obj,
                        Business_zones=business_zones_value,
                        defaults={
                            'Supply_volume': supply_volume,
                            'Bill_volume': bill_volume,
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
admin.site.register(business_zones_data, business_zones_admin)
