import numpy as np 
import torch as t 
import torch.nn as nn 

input_size = 28
timestep = 28
hidden_size = 64
batch_size = 100

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

# initialize trainable weights
kernel_f = t.randn(input_size, hidden_size, requires_grad=True, device=device)
kernel_i = t.randn(input_size, hidden_size, requires_grad=True, device=device)
kernel_o = t.randn(input_size, hidden_size, requires_grad=True, device=device)
kernel_c = t.randn(input_size, hidden_size, requires_grad=True, device=device)

recurrent_kernel_f = t.randn(hidden_size, hidden_size, requires_grad=True, device=device)
recurrent_kernel_i = t.randn(hidden_size, hidden_size, requires_grad=True, device=device)
recurrent_kernel_o = t.randn(hidden_size, hidden_size, requires_grad=True, device=device)
recurrent_kernel_c = t.randn(hidden_size, hidden_size, requires_grad=True, device=device)

bias_f = t.zeros(size=(hidden_size,), requires_grad=True, device=device)
bias_i = t.zeros(size=(hidden_size,), requires_grad=True, device=device)
bias_o = t.zeros(size=(hidden_size,), requires_grad=True, device=device)
bias_c = t.zeros(size=(hidden_size,), requires_grad=True, device=device)



dense_size = 10
dense_weight = t.randn(hidden_size, dense_size,requires_grad=True, device=device)
dense_bias = t.zeros(dense_size,requires_grad=True, device=device)




from torchvision import datasets, transforms
from torch.utils.data import DataLoader
train_dataset = datasets.MNIST(root = 'data/', train = True, 
                               transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'data/', train = False, 
                               transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_dataset, 
                        batch_size = batch_size, 
                        shuffle = True, 
                        num_workers=6,
                        pin_memory=True,
                        prefetch_factor=20,
                        
)
test_loader = DataLoader(dataset = test_dataset, batch_size= batch_size, shuffle = True)

optimizer = t.optim.Adam([
    kernel_f, kernel_i,kernel_o,kernel_c, 
    recurrent_kernel_f,recurrent_kernel_i,recurrent_kernel_o,recurrent_kernel_c,
    bias_f,bias_i,bias_o, bias_c, 
    dense_weight, dense_bias
], lr=0.01)

n_epochs = 20

for epc in range(n_epochs):

    batch_idx = 0
    for data in train_loader:
        inputs, labels = data 
        inputs = t.squeeze(inputs, axis=1)
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        # Re-initialize hidden state and cell state for every sample
        state_h = t.zeros(size=(batch_size, hidden_size), device=device)
        state_c = t.zeros(size=(batch_size, hidden_size), device=device)
        # Recurrent execution along the time axis
        for ts in range(timestep):
            input_t = inputs[:, ts, :]
            f_t = t.sigmoid(t.matmul(input_t, kernel_f) + t.matmul(state_h,recurrent_kernel_f) + bias_f)
            i_t = t.sigmoid(t.matmul(input_t, kernel_i) + t.matmul(state_h,recurrent_kernel_i) + bias_i)
            o_t = t.sigmoid(t.matmul(input_t, kernel_o) + t.matmul(state_h,recurrent_kernel_o) + bias_o)
            c_t_hat = t.tanh(t.matmul(input_t, kernel_c) + t.matmul(state_h, recurrent_kernel_c) + bias_c)
            state_c = t.mul(state_c, f_t) + t.mul(c_t_hat, i_t)
            state_h = t.mul(t.tanh(state_c), o_t)

        #Dense connection layer
        logits = t.matmul(state_h, dense_weight) + dense_bias

        # Loss calculation
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        batch_idx += 1

        if batch_idx % 100 == 0:
            print("epoch=%2d, loss = %f" % (epc, loss))

# Inference for the testing set
correct_count = 0
total_count = 0
for data in test_loader:
    inputs, labels = data 
    inputs = t.squeeze(inputs, axis=1)
    inputs, labels = inputs.cuda(), labels.cuda()

    state_h = t.zeros(size=(batch_size, hidden_size), device=device)
    state_c = t.zeros(size=(batch_size, hidden_size), device=device)
    for ts in range(timestep):
        input_t = inputs[:, ts, :]
        f_t = t.sigmoid(t.matmul(input_t, kernel_f) + t.matmul(state_h,recurrent_kernel_f) + bias_f)
        i_t = t.sigmoid(t.matmul(input_t, kernel_i) + t.matmul(state_h,recurrent_kernel_i) + bias_i)
        o_t = t.sigmoid(t.matmul(input_t, kernel_o) + t.matmul(state_h,recurrent_kernel_o) + bias_o)
        c_t_hat = t.tanh(t.matmul(input_t, kernel_c) + t.matmul(state_h, recurrent_kernel_c) + bias_c)
        state_c = t.mul(state_c, f_t) + t.mul(c_t_hat, i_t)
        state_h = t.mul(t.tanh(state_c), o_t)

    logits = t.matmul(state_h, dense_weight) + dense_bias
    sfm = nn.Softmax(dim=1)
    pred = t.argmax(sfm(logits), dim=1)

    correct = t.sum(pred == labels)
    correct_count += correct
    total_count += inputs.shape[0]
print("test set accuracy = %f" % (correct_count / total_count))

