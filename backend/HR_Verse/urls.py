from django.contrib import admin
from django.urls import path, include
from hr.views import home_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('hr.urls')),
    path('', home_view, name='home'),
]
