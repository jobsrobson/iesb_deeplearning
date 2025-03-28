# 🧠 1. Introdução ao PyTorch

## ✳️ O que é PyTorch?

PyTorch é um framework **open-source** para deep learning e computação numérica com foco 
]em **flexibilidade**, **performance** e **usabilidade**. Baseado em **Torch** (Lua), foi 
desenvolvido pelo **Facebook AI Research Lab (FAIR)** e lançado em **2016**.

---

# 🕰️ 2. Breve História e Motivações

| Ano  | Evento chave                                                     |
|------|------------------------------------------------------------------|
| 2011 | Torch (em Lua) utilizado por DeepMind e Facebook                 |
| 2015 | TensorFlow (Google) é lançado (substituindo o Theano)            |
| 2016 | Lançamento do PyTorch (baseado em Torch + Python)                |
| 2019 | PyTorch se torna padrão no FAIR e é adotado por MS, AWS          |
| 2022 | PyTorch é doado à Linux Foundation e se torna PyTorch Foundation |

### 🎯 Motivações do PyTorch:

- Simplicidade e leitura de código semelhante a **NumPy**
- **Dynamic computation graph (define-by-run)** — fácil de depurar
- Forte integração com o **ecossistema Python**
- Facilita **pesquisa e prototipagem**

---

# 📊 3. Comparativo PyTorch vs TensorFlow 2

| Critério              | PyTorch                        | TensorFlow 2                        |
|-----------------------|--------------------------------|-------------------------------------|
| Paradigma de execução | Dinâmico (define-by-run)       | Estático (Graph) + modo eager       |
| Sintaxe               | Python puro / NumPy-like       | API própria, mais verbosa           |
| Curva de aprendizado  | Rápida                         | Mais íngreme                        |
| Debugging             | Simples com `print()` ou `pdb` | Precisa de `tf.print` ou `tf.debug` |
| Comunidade acadêmica  | Muito forte                    | Mais adotado na indústria           |
| Ferramentas de deploy | TorchScript, ONNX              | TF Serving, TFLite                  |

---

# 🌐 4. PyTorch no mercado

- **Empresas que usam**: Facebook, Tesla, Microsoft, OpenAI, Amazon, NVIDIA, Hugging Face, Stability AI
- **Domina em pesquisa**: arXiv, NeurIPS, ICLR, ICML
- **Áreas de destaque**:
  - Visão computacional
  - Processamento de linguagem natural (NLP)
  - Modelos generativos (GANs, Diffusion)
  - **LLMs** (Large Language Models)

# 📦 Instalação
Para instalar o Pytorch usando o PIP, basta executar o comando

```
$ pip install torch torchvision
```

---

# 📦 O se deve saber sobre **Tensores**

## 🔍 O que são tensores?

Tensores são **estruturas de dados** fundamentais em Deep Learning — são a base de todo o processamento em bibliotecas como **PyTorch**, **TensorFlow** e outras.

> Em essência, um **tensor é uma generalização de matrizes e vetores** para múltiplas dimensões.

| Objeto Matemático | Representação Tensorial | Exemplo com PyTorch              |
|-------------------|-------------------------|----------------------------------|
| Escalar (número)  | Tensor 0D               | `torch.tensor(3.14)`             |
| Vetor             | Tensor 1D               | `torch.tensor([1.0, 2.0, 3.0])`  |
| Matriz            | Tensor 2D               | `torch.tensor([[1, 2], [3, 4]])` |
| Cubo de dados     | Tensor 3D               | `torch.randn(2, 3, 4)`           |
| Volume N-D        | Tensor N-dimensional    | `torch.randn(10, 3, 32, 32)`     |

---

## 🧠 Por que tensores são importantes?

- Eles **representam os dados de entrada e saída** de redes neurais (imagens, textos, séries temporais, etc).
- Todos os **parâmetros dos modelos (pesos e bias)** também são tensores.
- O **processo de backpropagation** (retropropagação) utiliza tensores e operações entre eles.
- São otimizados para **operações paralelas em GPU**.

---

## 📏 Importante: Tensores ≠ Arrays
Apesar de parecerem arrays do NumPy, tensores possuem vantagens importantes:

Autograd: registram operações para cálculo automático de derivadas

Compatibilidade com GPU

Operações otimizadas para Deep Learning


## ⚙️ Criando tensores com PyTorch

```python
import torch

# Tensor a partir de lista
a = torch.tensor([1.0, 2.0, 3.0])

# Tensor com valores aleatórios
b = torch.randn(2, 3)  # 2 linhas, 3 colunas

# Tensor com todos os zeros
z = torch.zeros(4, 4)

# Tensor com todos os uns
o = torch.ones(2, 2)
```
## 🧮 Operações com tensores

Tensores suportam **operações matemáticas** similares ao NumPy:

```python
import torch

x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])

print(x + y)        # Soma
print(x * y)        # Multiplicação, elemento a elemento
print(x @ y)        # Produto escalar
```


## 🚀 Tensores na GPU

Fundamental para aproveitar o paralelismo das GPUs, acelerando o treinamento de redes neurais.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1.0, 2.0], device=device)
```

## 📐 Forma (shape) e dimensão (rank)

```python
t = torch.randn(3, 4, 5)  # Tensor 3D

print(t.shape)    # (3, 4, 5)
print(t.ndim)     # 3 (dimensões)
```

| Termo	  | Significado                      |
|---------|----------------------------------|
| shape	  | Quantidade de elementos por eixo |
| ndim	   | Número de eixos (dimensões)      |
| size()	 | Alternativa a shape              |

# 🔁 Broadcasting
PyTorch permite operações entre tensores com formas diferentes, desde que sejam compatíveis:

```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([10, 20])

print(A + B)  # B é "expandido" automaticamente
```


---
