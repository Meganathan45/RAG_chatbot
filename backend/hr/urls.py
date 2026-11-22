from django.urls import path
from .views import employee_list_create, chat_view, health_check

urlpatterns = [
    path('employees/', employee_list_create, name='employee-list-create'),
    path('chat/', chat_view, name='chat'),
    path('health/', health_check, name='health-check'),
]
