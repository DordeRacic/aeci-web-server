from django.urls import path
from .views import uplaod_view

urlpatterns = [
    path("", upload_view, name="upload")
]