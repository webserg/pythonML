import matplotlib.pyplot as plt
import torch
import torch.nn as nn




fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

x = torch.arange(-5., 5., 0.1)
y = torch.sigmoid(x)
ax1.plot(x.numpy(), y.numpy())
ax1.set_title("sigmoid")

prelu = torch.nn.PReLU(num_parameters=1)
x = torch.arange(-5., 5., 0.1)
y = prelu(x)
ax2.plot(x.detach().numpy(), y.detach().numpy())
ax2.set_title("prelu")

x = torch.arange(-5., 5., 0.1)
y = torch.tanh(x)
ax3.plot(x.numpy(), y.numpy())
ax3.set_title("tanh")

relu = torch.nn.ReLU()
x = torch.arange(-5., 5., 0.1)
y = relu(x)
ax4.plot(x.numpy(), y.numpy())
ax4.set_title("relu")

plt.show()

softmax = nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y_output = softmax(x_input)
print(x_input)
print(y_output)
print(torch.sum(y_output, dim=1))

