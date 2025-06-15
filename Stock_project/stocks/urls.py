from django.urls import path
from . import views

urlpatterns = [
    path("api/history/<str:ticker>/", views.price_history, name="price_history"),
    path("api/stock/update/", views.update_stock, name="update_stock"),
    path("api/stock/data/", views.get_stock_data, name="get_stock_data"),
] 