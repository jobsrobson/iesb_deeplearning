import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

file_path = "/home/jobsr/Documents/GitHub/iesb_deeplearning/rnn/timeseries/data/CITIMON.csv"

try:
    # Carregar o CSV
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    print("CSV carregado")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    print(df.head())

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()


# Definir a coluna de interesse
column_of_interest = 'CONB'


# Usar 'DATE' como coluna de tempo e 'column_of_interest' como coluna de interesse
if df.shape[1] > 0:
    if column_of_interest in df.columns:
        series_name = column_of_interest
    else:
        print(f"Erro: A coluna de interesse '{column_of_interest}' não foi encontrada no CSV.")
        exit()
    # A coluna de tempo já foi usada como index ao ler o CSV (index_col=0), que é 'DATE'
else:
    print("Erro: O DataFrame está vazio ou não tem colunas.")
    exit()

print(f"\nUsando a coluna '{series_name}' como série temporal principal.")
main_series = df[series_name].values.astype(float)

# Verificar se a série contém valores NaN
if np.isnan(main_series).any():
    print("Detectados valores NaN. Realizando interpolação linear.")
    main_series = pd.Series(main_series).interpolate(method='linear').values

# Normalização dos dados: importante para RNNs
scaler = MinMaxScaler(feature_range=(-1, 1)) # -1 e 1 é comum para tanh
main_series_scaled = scaler.fit_transform(main_series.reshape(-1, 1)) # Reshape para (n_samples, n_features)

# 2. Preparação dos Dados para Treinamento (dados normalizados)
def create_dataset(series, input_sequence_length):
    X, y = [], []
    for i in range(len(series) - input_sequence_length):
        X.append(series[i : i + input_sequence_length])
        y.append(series[i + input_sequence_length])
    return np.array(X), np.array(y)

input_sequence_length = 30 # Comprimento da sequência
X_np, y_np = create_dataset(main_series_scaled.flatten(), input_sequence_length)

# Convertendo para tensores PyTorch
X_tensor = torch.from_numpy(X_np).float().unsqueeze(2)
y_tensor = torch.from_numpy(y_np).float().unsqueeze(1)

print(f"Formato de X_tensor: {X_tensor.shape}")
print(f"Formato de y_tensor: {y_tensor.shape}")

# Dividir em treino e teste (mantendo 80/20)
train_size = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# 3. Definição da Arquitetura da RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Parâmetros da RNN
input_size = 1
hidden_size = 64 
output_size = 1
num_layers = 2

# Instanciando o modelo
model = SimpleRNN(input_size, hidden_size, output_size, num_layers)

# Definindo a função de perda e o otimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # learning rate

# 4. Treinamento do Modelo
num_epochs = 300 # número de épocas
for epoch in range(num_epochs):
    model.train() # Coloca o modelo em modo de treinamento
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Clipagem de gradiente: Ajuda a evitar o problema de gradiente explosivo em RNNs
    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("Treinamento concluído!")

# 5. Avaliação e Previsão
model.eval()
with torch.no_grad():
    train_predict_scaled = model(X_train).numpy()
    test_predict_scaled = model(X_test).numpy()

# Inverter a normalização para plotar os resultados na escala original
train_predict = scaler.inverse_transform(train_predict_scaled)
y_train_original = scaler.inverse_transform(y_train.numpy())

test_predict = scaler.inverse_transform(test_predict_scaled)
y_test_original = scaler.inverse_transform(y_test.numpy())

# Plotagem dos resultados
# Precisamos alinhar as previsões com os dados originais no tempo
full_series_original = scaler.inverse_transform(main_series_scaled).flatten()

train_plot = np.empty_like(full_series_original)
train_plot[:] = np.nan
train_plot[input_sequence_length : len(train_predict) + input_sequence_length] = train_predict.flatten()

test_plot = np.empty_like(full_series_original)
test_plot[:] = np.nan
test_plot[len(train_predict) + input_sequence_length : len(train_predict) + input_sequence_length + len(test_predict)] = test_predict.flatten()

# EXIBIÇÃO DOS RESULTADOS DAS MÉTRICAS
print(f"Shape de train_predict: {train_predict.shape}")
print(f"Shape de test_predict: {test_predict.shape}")
print(f"Shape de y_train_original: {y_train_original.shape}")
print(f"Shape de y_test_original: {y_test_original.shape}")
print(f"Shape de full_series_original: {full_series_original.shape}")

# Plotando os resultados
plt.figure(figsize=(15, 7))
plt.plot(full_series_original, label=f'Dados Reais ({series_name})')
plt.plot(train_plot, label='Previsão Treino', alpha=0.7)
plt.plot(test_plot, label='Previsão Teste', alpha=0.7)
plt.title(f"Previsão de Série Temporal com RNN Simples - Dataset {series_name}")
plt.xlabel("Índice de Tempo")
plt.ylabel("Valor Original")
plt.legend()
plt.grid(True)
plt.show()
