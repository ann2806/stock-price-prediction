from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.admin import GroupAdmin as BaseGroupAdmin
from django.contrib.auth.models import User, Group
from .models import StockData

@admin.register(StockData)
class StockDataAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume')
    search_fields = ('symbol', 'date')
    list_filter = ('symbol', 'date')
    ordering = ('-date',)  # Show latest records first
    date_hierarchy = 'date'  # Adds a date filter navigation
    list_per_page = 10

    fieldsets = (
        ("Stock Information", {
            "fields": ("symbol", "date")
        }),
        ("Price Details", {
            "fields": ("open_price", "high_price", "low_price", "close_price")
        }),
        ("Trading Data", {
            "fields": ("volume",)
        }),
    )
