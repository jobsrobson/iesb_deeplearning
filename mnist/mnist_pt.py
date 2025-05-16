from typing import Dict, Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

# Cria a classe Net que representa a rede neural, herdando de nn.Module
# Define a arquitetura da rede neural, que consiste em duas camadas densas (fully connected)
# e uma função de ativação ReLU entre elas.
# A primeira camada tem 784 entradas (tamanho da imagem MNIST) e 512 saídas.
# A segunda camada tem 512 entradas e 10 saídas (uma para cada dígito de 0 a 9).
# A função de ativação ReLU é aplicada após a primeira camada.
# A função de perda utilizada é a CrossEntropyLoss, que é adequada para problemas de classificação.
# O otimizador utilizado é o SGD (Stochastic Gradient Descent) com uma taxa de aprendizado de 0.001.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(784, 512)
        self.dense2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    # Ainda dentro da classe, definimos o método forward, que é responsável por definir a passagem
    # dos dados pela rede neural. Ele recebe um tensor de entrada x, aplica a primeira camada densa,
    # seguida pela função de ativação ReLU, e depois aplica a segunda camada densa.
    # O resultado final é retornado como a saída da rede neural.
    # O método forward é chamado automaticamente quando chamamos a instância da classe Net com um tensor de entrada.
    # Isso é feito através do método __call__ da classe nn.Module, que chama o método forward.
    # O método forward é onde a mágica acontece, pois ele define como os dados fluem através da rede.
    # Ele aplica as operações definidas na classe Net, como as camadas densas e a função de ativação.

    def forward(self, x):
        x = self.dense1(x)      # Aplicando a primeira camada densa - entrada
        x = self.relu(x)        # Aplicando a função de ativação ReLU - ativação
        x = self.dense2(x)      # Aplicando a segunda camada densa - saída
        return x

# A função train_and_log_step é responsável por treinar o modelo e registrar os resultados no MLflow.
# Ela recebe um dicionário de parâmetros que contém o tamanho do lote (batch size), a taxa de aprendizado
# (learning rate) e o número de épocas (epochs) para o treinamento.
# Dentro dessa função, os dados são carregados e preparados para o treinamento.
# O conjunto de dados MNIST é utilizado, que contém imagens de dígitos manuscritos.
# As imagens são transformadas em tensores e redimensionadas para um vetor de 784 elementos (28x28).
# O modelo é criado a partir da classe Net, e a função de perda e o otimizador são definidos.
# O treinamento é realizado em um loop que itera sobre as épocas e os lotes de dados.
# Durante o treinamento, a perda média é calculada e registrada no MLflow.
# Após o treinamento, o modelo é avaliado no conjunto de teste e a perda média e a acurácia são registradas.
# O modelo treinado é salvo como um artefato no MLflow.

def train_and_log_step(params: Dict):
    batch_size: int = params['batch_size']
    epochs: int = params['epochs']
    learning_rate: float = params['learning_rate']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1 )),
    ])

    # Carregar e preparar os dados
    train_loader: DataLoader[Any] = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader: DataLoader[Any] = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True,
    )

    # Configuração do dispositivo (GPU ou CPU)
    # Verifica se há GPU disponível e, se sim, usa-a; caso contrário, usa a CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net().to(device)
    criterion: CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    with mlflow.start_run(run_name='mnist_train'):
        mlflow.log_params(params)

        # Treinamento do modelo, através de um loop que itera sobre as épocas e os lotes de dados.
        for epoch in range(epochs):
            model.train()
            running_loss: float = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                # Logits são as saídas da rede neural antes de aplicar a função de ativação softmax (padrão do PyTorch).
                logits = model(x_batch)
                # Calcula a Loss (perda) entre as previsões do modelo e os rótulos reais.
                # A função de perda CrossEntropyLoss é utilizada para calcular a diferença entre as previsões e os rótulos.
                loss: CrossEntropyLoss = criterion(logits, y_batch)
                # Faz backpropagation para calcular os gradientes.
                # O otimizador atualiza os pesos do modelo com base nos gradientes calculados.
                loss.backward()
                # O otimizador aplica os gradientes calculados para atualizar os pesos do modelo.
                optimizer.step()
                # A perda média é acumulada para calcular a perda total do lote.
                running_loss += loss.item()

            # Calcula a perda média para a época atual
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}], Loss: {avg_loss:.4f}')

        # Model.eval() coloca o modelo em modo de avaliação, desativando o dropout e a normalização em lote.
        model.eval()

        # Zeramos as variáveis para calcular a acurácia e a perda média no conjunto de teste
        correct = 0
        total = 0
        test_loss = 0.0

        # Avaliação do modelo no conjunto de teste
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                test_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        # Calcula a acurácia e a perda média no conjunto de teste
        test_acc = correct / total
        avg_test_loss = test_loss / len(test_loader)
        mlflow.log_metric("test_loss", avg_test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.4f}")

        # Log do modelo no MLflow
        mlflow.pytorch.log_model(model, "model")

# Parâmetros de configuração
# Define os parâmetros de configuração do modelo, como o tamanho do lote (batch size),
# a taxa de aprendizado (learning rate) e o número de épocas (epochs).
params = {
    "batch_size": 64,
    "learning_rate": 0.01,
    "epochs": 5
}


# O bloco if __name__ == "__main__": é utilizado para garantir que o código dentro dele
# seja executado apenas quando o script é executado diretamente, e não quando é importado como um módulo.
# Isso é útil para evitar a execução de código indesejado quando o script é importado em outro lugar.
# Dentro desse bloco, chamamos a função train_and_log_step com os parâmetros definidos acima.
# Isso inicia o treinamento do modelo e registra os resultados no MLflow.
if __name__ == "__main__":
    train_and_log_step(params)