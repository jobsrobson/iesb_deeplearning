import torch

def compute_z(a,b,c): # z = 2(a-b) + c
    r1 = torch.sub(a, b)
    r2 = torch.mul(2.0, r1)
    z = torch.add(r2, c)
    return z


# Exemplo de uso com tensor de duas dimensões
if __name__ == "__main__":
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[2.0, 3.0]])
    c = torch.tensor([[3.0, 4.0]])
    
    z = compute_z(a, b, c)
    print("Resultado de z:")
    print(z)

