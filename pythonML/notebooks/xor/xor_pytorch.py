import numpy as np
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# the four different states of the XOR gate
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

# the four expected results in the same order
target_data = np.array([[0], [1], [1], [0]], "float32")

input_size = 2
output_size = 1
num_epochs = 1001
learning_rate = 0.05


class XOR_NET(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        self.output = nn.Linear(8, 1)
        torch.nn.init.xavier_uniform_(self.output.weight)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            # m.weight.data.normal_(0, 1)
            print(m.weight)


def mse_loss(output, target):
    l = torch.mean((output - target) ** 2)
    return l


model = XOR_NET()
weights_init(model)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
loss_fn = nn.MSELoss()
# loss_fn = mse_loss

loss_history = []
epoch_history = []
# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(training_data)
    targets = torch.from_numpy(target_data)

    # Forward pass
    optimizer.zero_grad()
    y_hat = model(inputs)
    loss = loss_fn.forward(input=y_hat, target=targets)
    # Backward and optimize
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        loss_history.append(loss.item())
        epoch_history.append(epoch)

# Plot the graph
# predicted = model(torch.from_numpy(training_data)).detach().numpy()
plt.plot(epoch_history, loss_history, 'ro', label='Original data')
# plt.plot(training_data, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'pytorch_xor_model.ckpt')

with torch.no_grad():
    res = model(torch.from_numpy(training_data))
torch.set_printoptions(precision=2)
print(res)

print(model.state_dict())
