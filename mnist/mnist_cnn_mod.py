import torch
import torch.nn as nn
import yaml # Embora o carregamento de YAML tenha sido substituído por um dict, a importação foi mantida do original.
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
import os
from datetime import datetime

# Convolutional Neural Network (CNN) for MNIST - MODIFICADA
class MNISTCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, conv1_filters=32, conv2_filters=64, num_dense_neurons=128):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # A entrada da camada densa depende de conv2_filters
        # Imagem 28x28 -> conv1 (padding=1) -> 28x28 -> max_pool (2,2) -> 14x14
        # -> conv2 (padding=1) -> 14x14 -> max_pool (2,2) -> 7x7
        # Então, o tamanho achatado é conv2_filters * 7 * 7
        self.fc1_input_features = conv2_filters * 7 * 7
        self.dense1 = nn.Linear(self.fc1_input_features, num_dense_neurons)
        self.dropout = nn.Dropout(dropout_rate) # Camada de Dropout
        self.output_fc = nn.Linear(num_dense_neurons, 10) # Camada de saída

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x) # Aplicando Dropout
        x = self.output_fc(x)
        return x


def mnist_uploader():
    # Nota: 'transform' e 'batch_size' são usados aqui, mas são definidos
    # no escopo global do bloco if __name__ == "__main__".
    _datasets = {
        "train_dataset": datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        ),
        "test_dataset": datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        ),
    }
    _data_loaders = {
        "train_loader": DataLoader(
            _datasets["train_dataset"], batch_size=batch_size, shuffle=True
        ),
        "test_loader": DataLoader(
            _datasets["test_dataset"], batch_size=batch_size, shuffle=False
        ),
    }
    return _datasets, _data_loaders


def train_model(model, train_loader, criterion, optimizer, epochs, current_device):
    final_epoch_loss = 0.0 # Para armazenar a perda da última época
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(current_device), target.to(current_device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        final_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {final_epoch_loss:.4f}")
    return final_epoch_loss # Retorna a perda da última época


# NOVA FUNÇÃO para salvar resultados em CSV
def save_results_to_csv(filepath, params_dict, metrics_dict):
    file_exists = os.path.isfile(filepath)
    is_empty = os.path.getsize(filepath) == 0 if file_exists else True

    # Combinar os dicionários para facilitar a escrita e adicionar timestamp
    data_row = {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    data_row.update(params_dict)
    data_row.update(metrics_dict)
    
    # Definir a ordem das colunas explicitamente para consistência
    # Inclui os novos parâmetros da arquitetura
    ordered_fieldnames = [
        'timestamp', 'batch_size', 'epochs', 'learning_rate', 'momentum', 
        'dropout_rate', 'conv1_filters', 'conv2_filters', 'num_dense_neurons',
        'final_train_loss', 'test_accuracy'
    ]
    
    # Filtrar data_row para incluir apenas as chaves em ordered_fieldnames e na ordem correta
    # Se uma chave não estiver em data_row, seu valor será None (comportamento de .get())
    row_to_write = {field: data_row.get(field) for field in ordered_fieldnames}

    try:
        with open(filepath, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ordered_fieldnames)
            if not file_exists or is_empty: # Checa se o arquivo é novo ou vazio
                writer.writeheader()
            writer.writerow(row_to_write)
        print(f"Resultados salvos em {filepath}")
    except IOError:
        print(f"Erro ao escrever no arquivo CSV: {filepath}")


# ============================================= #

if __name__ == "__main__":
    # Configurações da execução (substituindo o carregamento de YAML por um dict para este exemplo)
    config = {
        "batch_size": 64,
        "epochs": 5,  # Reduzido para execução mais rápida de exemplo
        "learning_rate": 0.01,
        "momentum": 0.9,
        "dropout_rate": 0.25,     # Novo parâmetro da arquitetura
        "conv1_filters": 16,      # Novo parâmetro da arquitetura
        "conv2_filters": 32,      # Novo parâmetro da arquitetura
        "num_dense_neurons": 64   # Novo parâmetro da arquitetura
    }

    # Extraindo hiperparâmetros do dicionário 'config'.
    batch_size = config["batch_size"] # Usado globalmente por mnist_uploader
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    dropout_rate = config["dropout_rate"]
    conv1_filters = config["conv1_filters"]
    conv2_filters = config["conv2_filters"]
    num_dense_neurons = config["num_dense_neurons"]


    # Configurando o dispositivo (GPU se disponível, senão CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    # Definindo as transformações de imagem.
    transform = transforms.Compose( # Usado globalmente por mnist_uploader
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    # Instanciando o modelo com os novos parâmetros e movendo-o para o dispositivo.
    model = MNISTCNN(
        dropout_rate=dropout_rate,
        conv1_filters=conv1_filters,
        conv2_filters=conv2_filters,
        num_dense_neurons=num_dense_neurons
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )

    ds, dl = mnist_uploader()

    # Treinando o modelo
    # Passando 'device' explicitamente para a função de treino
    final_train_loss = train_model(model, dl["train_loader"], criterion, optimizer, epochs, device)

    # Avaliando o modelo no conjunto de teste.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch, target_batch in dl["test_loader"]: # Renomeado para evitar conflito com 'data' global
            data_batch, target_batch = data_batch.to(device), target_batch.to(device)
            output = model(data_batch)
            _, predicted = torch.max(output.data, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum().item()
    
    test_accuracy = (100 * correct / total) if total > 0 else 0.0
    print(f"Accuracy: {test_accuracy:.2f}%")

    # Preparando dados para salvar no CSV
    params_to_save = config.copy() # Usamos o dicionário config original para os parâmetros
    
    metrics_to_save = {
        "final_train_loss": final_train_loss,
        "test_accuracy": test_accuracy
    }
    
    # Salvando os resultados
    csv_filepath = "mnist_training_runs.csv"
    save_results_to_csv(csv_filepath, params_to_save, metrics_to_save)