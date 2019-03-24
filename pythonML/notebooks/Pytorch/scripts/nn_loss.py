import torch
import torch.nn as nn
import torch.nn.functional as F

def myCrossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]
    # batch_size
    tmp_outputs = F.softmax(outputs, dim=1)
    print(tmp_outputs)# compute the log of softmax values
    outputs = F.log_softmax(outputs, dim=1)
    print(outputs)# compute the log of softmax values
    outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
    return -torch.sum(outputs)/len(labels)

m = nn.LogSoftmax()
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = torch.randn(3, 5)
print(input)
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
print(len(target))
output = loss(m(input), target)
print(output)
print(output.item())
output2 = myCrossEntropyLoss(input, target)
print(output2)
#Mean Squared Error Loss
mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
print(outputs)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)
#Categorical Cross-Entropy Loss
ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([1, 0, 3], dtype=torch.int64)
loss = ce_loss(outputs, targets)
print(loss)

