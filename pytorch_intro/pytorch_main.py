import torch

def sum_tensor(a, b):
    return a + b


def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    print(sum_tensor(a, b))
    print(set_device())
