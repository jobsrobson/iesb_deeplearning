import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Convolutional Neural Network (CNN) for MNIST


class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__() # Chama o construtor da classe pai nn.Module.

        # Camada Convolucional 1
        # Entrada: 1 canal (imagem MNIST é em tons de cinza)
        # Saída: 32 canais (32 mapas de características)
        # kernel_size=3: filtro de convolução 3x3
        # padding=1: adiciona uma borda de 1 pixel ao redor da entrada.
        #   Isso ajuda a manter o tamanho espacial da saída igual ao da entrada
        #   após a convolução com um kernel 3x3 e stride 1.
        #   (W - K + 2P)/S + 1 = (28 - 3 + 2*1)/1 + 1 = 28. A imagem 28x28 continua 28x28.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Camada Convolucional 2
        # Entrada: 32 canais (da saída da conv1)
        # Saída: 64 canais
        # kernel_size=3, padding=1: mesma lógica da conv1.
        #   A entrada aqui (após o max_pool da conv1) será 14x14. Após esta conv, continuará 14x14.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Função de Ativação ReLU (Rectified Linear Unit)
        # f(x) = max(0, x). Introduz não-linearidade.
        self.relu = nn.ReLU()

        # Camada de Max Pooling
        # kernel_size=2, stride=2: Janela de 2x2, move de 2 em 2 pixels.
        # Reduz a dimensionalidade espacial pela metade (ex: 28x28 -> 14x14).
        self.max_pool = nn.MaxPool2d(2, 2)

        # Camada de Flatten (Achatamento)
        # Transforma o tensor multidimensional (mapas de características) em um vetor 1D
        # para que possa ser processado por camadas densas (Lineares).
        self.flatten = nn.Flatten()

        # Camada Densa (Totalmente Conectada ou Linear)
        # Entrada: 64 * 7 * 7. Vamos entender esse número:
        #   - Imagem original: 28x28
        #   - Após conv1 (28x28) + relu + max_pool (14x14)
        #   - Após conv2 (14x14) + relu + max_pool (7x7)
        #   - A saída da conv2 tem 64 canais. Então, temos 64 mapas de 7x7.
        #   - Total de neurônios após achatamento: 64 * 7 * 7 = 3136.
        # Saída: 128 neurônios.
        self.dense = nn.Linear(64 * 7 * 7, 128)

        # Camada de Saída (Densa)
        # Entrada: 128 neurônios (da camada densa anterior)
        # Saída: 10 neurônios (um para cada classe do MNIST: 0 a 9).
        self.output = nn.Linear(128, 10)


    def forward(self, x): # Define a passagem para frente (forward pass) dos dados pela rede.
        # x: tensor de entrada (lote de imagens)

        # Bloco Convolucional 1
        x = self.conv1(x)      # Aplica a primeira convolução
        x = self.relu(x)       # Aplica ReLU
        x = self.max_pool(x)   # Aplica Max Pooling

        # Bloco Convolucional 2
        x = self.conv2(x)      # Aplica a segunda convolução
        x = self.relu(x)       # Aplica ReLU
        x = self.max_pool(x)   # Aplica Max Pooling

        # Achatamento e Camadas Densas
        x = self.flatten(x)    # Achata o tensor
        x = self.dense(x)      # Aplica a primeira camada densa
        x = self.relu(x)       # Aplica ReLU novamente (comum após camadas densas, exceto a última de saída para classificação)
        x = self.output(x)     # Aplica a camada de saída, produzindo os logits.

        return x # Retorna os logits (saídas brutas antes da Softmax)




def mnist_uploader():
    # Nota: 'transform' e 'batch_size' são usados aqui, mas são definidos
    # no escopo global do bloco if __name__ == "__main__".
    # Isso significa que esta função não é totalmente independente e depende
    # dessas variáveis globais para funcionar como está.

    _datasets = {
        "train_dataset": datasets.MNIST( # Carrega o conjunto de TREINO do MNIST.
            root="./data",       # Diretório onde os dados serão baixados/armazenados.
            train=True,          # Indica que é o conjunto de treino.
            download=True,       # Baixa o dataset se não estiver presente.
            transform=transform  # Aplica as transformações definidas globalmente.
        ),
        "test_dataset": datasets.MNIST( # Carrega o conjunto de TESTE do MNIST.
            root="./data",
            train=False,         # Indica que é o conjunto de teste.
            download=True,
            transform=transform
        ),
    }
    _data_loaders = {
        "train_loader": DataLoader( # Cria um DataLoader para o conjunto de treino.
            _datasets["train_dataset"], # O dataset a ser carregado.
            batch_size=batch_size,      # Tamanho do lote (definido globalmente).
            shuffle=True                # Embaralha os dados a cada época (bom para treino).
        ),
        "test_loader": DataLoader( # Cria um DataLoader para o conjunto de teste.
            _datasets["test_dataset"],
            batch_size=batch_size,
            shuffle=False               # Não é necessário embaralhar para teste/validação.
        ),
    }
    return _datasets, _data_loaders # Retorna os datasets e os data loaders.



def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs): # Loop através do número de épocas especificado.
        model.train() # Coloca o modelo em modo de treinamento.
                      # Ativa camadas como Dropout e BatchNorm (se existirem).
        total_loss = 0 # Acumulador para a perda total na época.

        # Loop através dos lotes (batches) de dados de treino.
        # batch_idx é o índice do lote, (data, target) são as imagens e seus rótulos.
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move os dados e rótulos para o dispositivo configurado (GPU ou CPU).
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() # Zera os gradientes acumulados de iterações anteriores. Essencial!

            output = model(data) # Passagem para frente (forward pass): obtém as previsões do modelo.
            loss = criterion(output, target) # Calcula a perda entre as previsões e os rótulos verdadeiros.

            loss.backward() # Passagem para trás (backward pass): calcula os gradientes da perda.
            optimizer.step() # Atualiza os pesos do modelo usando os gradientes.

            total_loss += loss.item() # Acumula a perda do lote. '.item()' extrai o valor escalar.

        # Imprime a perda média da época.
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


# ============================================= #

if __name__ == "__main__":
    # Carregando configurações do arquivo YAML
    config = {
        "batch_size": 64,
        "epochs": 10,
        "learning_rate": 0.01,
        "momentum": 0.9
    }

    # Extraindo hiperparâmetros do dicionário 'config'.
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    momentum = config["momentum"] # Usado pelo otimizador SGD.

    # Configurando o dispositivo (GPU se disponível, senão CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    # ============================================= #

    # Definindo as transformações de imagem.
    transform = transforms.Compose( # Agrupa múltiplas transformações.
        [
            transforms.ToTensor(), # Converte a imagem (PIL/NumPy) para um Tensor PyTorch
                                   # e normaliza os valores dos pixels para o intervalo [0.0, 1.0].
            transforms.Normalize((0.1307,), (0.3081,)) # Normaliza o tensor com a média (0.1307)
                                                      # e o desvio padrão (0.3081) do dataset MNIST.
                                                      # Os valores são tuplas pois poderiam ser para múltiplos canais (ex: (mean_r, mean_g, mean_b)).
        ]
    )

    # Instanciando o modelo e movendo-o para o dispositivo.
    model = MNISTCNN().to(device)

    # Definindo a função de perda (Criterion).
    # CrossEntropyLoss é comumente usada para problemas de classificação multi-classe.
    # Ela combina LogSoftmax e NLLLoss.
    criterion = nn.CrossEntropyLoss()

    # Definindo o otimizador.
    # SGD (Stochastic Gradient Descent) com momento.
    optimizer = torch.optim.SGD(
        model.parameters(), # Parâmetros do modelo a serem otimizados.
        lr=learning_rate,   # Taxa de aprendizado.
        momentum=momentum   # Fator de momento (ajuda a acelerar SGD na direção relevante e amortece oscilações).
    )

    # Carregando os datasets e data loaders.
    # ds contém os datasets, dl contém os data loaders.
    ds, dl = mnist_uploader()
    # print(ds["train_dataset"][0]) # Linha comentada para inspecionar uma amostra.

    # Treinando o modelo.
    train_model(model, dl["train_loader"], criterion, optimizer, epochs)

    # Avaliando o modelo no conjunto de teste.
    model.eval() # Coloca o modelo em modo de avaliação.
                 # Desativa camadas como Dropout e BatchNorm (se existirem) para que a inferência seja determinística.
    correct = 0 # Contador de previsões corretas.
    total = 0   # Contador do total de amostras de teste.

    # `torch.no_grad()` desabilita o cálculo de gradientes durante a avaliação,
    # o que economiza memória e acelera o processo, já que não faremos backpropagation.
    with torch.no_grad():
        for data, target in dl["test_loader"]: # Itera sobre os lotes do conjunto de teste.
            data, target = data.to(device), target.to(device) # Move dados para o dispositivo.
            output = model(data) # Obtém as previsões (logits) do modelo.

            # `torch.max(output.data, 1)` retorna os valores máximos e seus índices ao longo da dimensão 1 (dimensão das classes).
            # `_` recebe os valores máximos (que não usamos aqui).
            # `predicted` recebe os índices das classes com maior probabilidade (as classes preditas).
            _, predicted = torch.max(output.data, 1)

            total += target.size(0) # Adiciona o número de amostras no lote ao total.
            correct += (predicted == target).sum().item() # Compara previsões com rótulos verdadeiros,
                                                         # soma as corretas e converte para um número Python.

    # Imprime a acurácia final no conjunto de teste.
    print(f"Accuracy: {100 * correct / total:.2f}%")


    # Para que este código funcione, você precisaria de um arquivo chamado config_model.yaml no mesmo diretório, com o seguinte conteúdo (por exemplo):
    # batch_size: 64
    # epochs: 10
    # learning_rate: 0.01
    # momentum: 0.9

