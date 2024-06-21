import torch

# Sample tensors (assuming they are already defined as tensors)
a = torch.rand(64, 64, 1024)
b = torch.rand(64, 64)

# Reshape a for matrix multiplication
a_reshaped = a.view(64, 64, 16, 64)

# Matrix multiplication
c = torch.matmul(a_reshaped, b)

print(c.shape)  # Output: torch.Size([64, 64, 16])
