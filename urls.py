from django.urls import path
from .views import predict_stock

urlpatterns = [
    path("predict-stock/", predict_stock, name="predict_stock"),
]

