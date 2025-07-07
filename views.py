from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .utils import (
    load_and_process_data, scale_and_split_data, load_saved_model, train_linear_regression,
    train_ann, plot_results,predict_future_prices,train_lstm,train_cnn,plot_future_predictions,plot_existing_predictions
)
from sklearn.metrics import mean_absolute_percentage_error as mape
from .models import StockData

@csrf_exempt
def predict_stock(request):
    if request.method == "POST":
        stock_symbol = request.POST.get("stock_symbol")
        model_choice = request.POST.get("model")
        
        df = load_and_process_data(stock_symbol)
        if df is None:
            return JsonResponse({"error": "No data found for the selected stock symbol"}, status=400)
        
        (X_train, X_test, y_train, y_test), scaler = scale_and_split_data(df)
        
        models = {
            "linear_regression": train_linear_regression,
            "ann": train_ann,
            "lstm": train_lstm,
            "cnn": train_cnn
        }
        
        model = load_saved_model(stock_symbol, model_choice)
        if model:
            print(f"✅ Loaded saved {model_choice} model for {stock_symbol}")
            y_pred = model.predict(X_test.reshape(X_test.shape[0], -1)) if model_choice == "linear_regression" else model.predict(X_test).flatten()
            accuracy = 100 - (mape(y_test, y_pred) * 100)
        else:
            print(f"🚀 Training new {model_choice} model for {stock_symbol}")
            model, y_pred, accuracy = models[model_choice](X_train, X_test, y_train, y_test, stock_symbol)
        
        # Get both actual and predicted prices
        future_prices = predict_future_prices(model, df, X_test[-1], scaler, future_days=5)
        
        return JsonResponse({
            "stock_symbol": stock_symbol,
            "model": model_choice,
            "accuracy": round(accuracy, 2),
            "test_plot": plot_results(y_test, y_pred, scaler, model_choice),
            "plot_existing_predictions": plot_existing_predictions(df, future_prices["predicted_prices"], model_choice, 5),
            "future_plot": plot_future_predictions(df, future_prices["predicted_prices"], model_choice),
            "compare_prices": future_prices,
        })
    
    return render(request, "stock_prediction.html")
