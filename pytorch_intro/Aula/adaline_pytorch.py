# Atividade Prática
# Converter o código de Adaline em Numpy para PyTorch
# Arquivo Numpy: adaline/adaline_np.py

import torch
import torch.nn as nn


# Função de Ativação
def activation_function(x: torch.Tensor) -> torch.Tensor:
    return x

# Inicialização dos Pesos
def initialize_weights(n_features: int) -> torch.Tensor:
    return torch.randn(n_features + 1) * 0.01

# Função de Predição
def predict(X: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    X_bias = torch.cat((torch.ones(X.shape[0], 1), X), dim=1)
    return activation_function(X_bias @ weights)

# Função de Erro Quadrático Médio
def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.mean((y_true - y_pred) ** 2)


# Função de Atualização dos Pesos
def update_weights(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, lr: float) -> torch.Tensor:
    X_bias = torch.cat((torch.ones(X.shape[0], 1), X), dim=1)
    y_pred = predict(X, weights)
    gradient = -2 * (X_bias.T @ (y - y_pred)) / X.shape[0]
    return weights - lr * gradient

# Função de Treinamento
def train(X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 0.01) -> torch.Tensor:
    weights = initialize_weights(X.shape[1])
    for _ in range(epochs):
        weights = update_weights(X, y, weights, lr)
    return weights


    