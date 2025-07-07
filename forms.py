from django import forms
from prediction_app.models import StockData  # Ensure your StockData model exists

class StockPredictionForm(forms.Form):
    stock_symbol = forms.ModelChoiceField(
        queryset=StockData.objects.values_list("symbol", flat=True).distinct(),
        label="Select Stock Symbol",
        widget=forms.Select(attrs={"class": "form-control select2"}),
    )
