# utils/metrics.py
import numpy as np

def RSE(pred, true):
    """Relative Squared Error"""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    """Correlation Coefficient"""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    # Tránh chia cho 0
    d += 1e-5
    return (u / d).mean(-1)

def MAE(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    """Mean Squared Error"""
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """Mean Absolute Percentage Error (Có xử lý chia cho 0)"""
    return np.mean(np.abs((pred - true) / (true + 1e-5)))

def MSPE(pred, true):
    """Mean Squared Percentage Error (Có xử lý chia cho 0)"""
    return np.mean(np.square((pred - true) / (true + 1e-5)))

def metric(pred, true):
    """
    Hàm tổng hợp tính toán toàn bộ metrics quan trọng.
    Lưu ý: pred và true nên được đưa về đơn vị thực tế (km, m/s, hPa) 
    trước khi tính toán để có ý nghĩa vật lý.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr