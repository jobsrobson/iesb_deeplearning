# Teste para o pytorch_grafo
import torch
import unittest
from pytorch_grafo.pytorch_grafo import compute_z

class TestComputeZ(unittest.TestCase):
    def test_compute_z(self):
        a = torch.tensor(1.0)
        b = torch.tensor(2.0)
        c = torch.tensor(3.0)
        expected_z = 2 * (a - b) + c
        result = compute_z(a, b, c)
        self.assertAlmostEqual(result.item(), expected_z.item(), places=6)

        a = torch.tensor(-1.0)
        b = torch.tensor(-2.0)
        c = torch.tensor(-3.0)
        expected_z = 2 * (a - b) + c
        result = compute_z(a, b, c)
        self.assertAlmostEqual(result.item(), expected_z.item(), places=6)

        a = torch.tensor(0.0)
        b = torch.tensor(0.0)
        c = torch.tensor(0.0)
        expected_z = 2 * (a - b) + c
        result = compute_z(a, b, c)
        self.assertAlmostEqual(result.item(), expected_z.item(), places=6)

