import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    计算 MAPE，自动剔除真实值(true)为 0 的样本，避免除以零错误。
    适用于光伏预测（剔除夜间数据）。
    """
    # 1. 创建掩码
    mask = true > 0.5
    
    # 2. 安全检查：如果全都是 0（例如全是晚上的数据），直接返回 0
    if np.sum(mask) == 0:
        return 0.0
    
    # 3. 只利用掩码筛选出的数据进行计算
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
