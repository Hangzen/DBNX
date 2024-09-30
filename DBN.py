import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPULICATE_LIB_OK']='TRUE'
data = pd.read_excel('PRS.xlsx', sheet_name='Sheet1')

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.nn.init.xavier_uniform_(torch.randn(n_visible, n_hidden)))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        h = (torch.rand_like(h_prob) < h_prob).float()
        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return v_prob, h_prob

    def encode(self, x):
        if x.shape[1] != self.W.shape[0]:
            x = x.view(-1, self.W.shape[0])
        encode_data = torch.sigmoid(torch.matmul(x, self.W) + self.h_bias)
        return encode_data

    def gibbs_sampling(self, v, k=1):
        for _ in range(k):
            _, h = self.forward(v)
            v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
            v = (torch.rand_like(v_prob) < v_prob).float()
        return v
    def energy(self, v, h):
        energy = -torch.sum(self.v_bias * v) - torch.sum(self.h_bias * h) - torch.sum(torch.matmul(v, self.W) * h)
        return energy

    def contrastive_divergence(self, v, lr):
        v_0 = v
        v_k, h_k = self.forward(v_0)
        x_k = self.gibbs_sampling(v_0)
        v_k_prime, h_k_prime = self.forward(x_k)

        log_likelihood_loss = torch.sum(v_0 * torch.log(v_k_prime) + (1 - v_0) * torch.log(1 - v_k_prime))

        energy_v_0 = self.energy(v_0, h_k)
        energy_x_k = self.energy(x_k, h_k_prime)
        energy_difference = energy_v_0 - energy_x_k

        total_loss = -log_likelihood_loss + energy_difference

        delta_W = torch.matmul(v_0.t(), h_k) - torch.matmul(x_k.t(), h_k_prime)
        delta_v_bias = torch.sum(v_0 - x_k, dim=0)
        delta_h_bias = torch.sum(h_k - h_k_prime, dim=0)
        self.W.grad = (delta_W / v_0.shape[0])
        self.v_bias.grad = delta_v_bias / v_0.shape[0]
        self.h_bias.grad = delta_h_bias / v_0.shape[0]
        self.W.data += lr * self.W.grad
        self.v_bias.data += lr * self.v_bias.grad
        self.h_bias.data += lr * self.h_bias.grad

        return total_loss

class DBN(nn.Module):
    def __init__(self, num_layers):
        super(DBN, self).__init__()
        self.rbm_layers = nn.ModuleList()
        self.num_layers = num_layers

    def forward(self, x):
        encoded_data = x
        for rbm_layer in self.rbm_layers:
            encoded_data = rbm_layer.encode(encoded_data)
        return encoded_data

    def compute_hidden_units(self, weight_matrix):
        U, S, Vt = np.linalg.svd(weight_matrix)
        rank = np.sum(S > 1e-10)
        total_singular_values_sum = np.sum(S)
        cumulative_sum = np.cumsum(S) / total_singular_values_sum
        new_m_values = []
        for new_m in range(1, rank + 1):
            if cumulative_sum[new_m - 1] >= 0.95:
                new_m_values.append(new_m)
                break
        m = new_m_values[-1]
        if len(self.rbm_layers) + 1 == self.num_layers and m > 1:
            m = 1

        return m

    def trainy(self, data, num_epochs, batch_size, lr=0.01):
        original_data = data.clone()
        print(f"Training Layer 1...")
        n_visible_units = data.shape[1]
        if self.num_layers == 1:
            n_hidden_units = 1
        else:
            n_hidden_units = data.shape[1]
        rbm_layer_1 = RBM(n_visible_units, n_hidden_units)
        self.rbm_layers.append(rbm_layer_1)
        optimizer = torch.optim.Adam(rbm_layer_1.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            total_loss = 0.0
            for j in range(0, data.shape[0], batch_size):
                batch_data = data[j:j + batch_size]
                v = batch_data.requires_grad_()
                loss = rbm_layer_1.contrastive_divergence(v, lr)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / data.shape[0]}")

        for i in range(1, self.num_layers):
            print(f"Training Layer {i + 1}...")
            previous_layer = self.rbm_layers[i - 1]
            for param in previous_layer.parameters():
                param.requires_grad = False
            data = previous_layer.encode(data)
            weight_matrix = previous_layer.W.detach().numpy()
            n_visible_units = data.shape[1]
            n_hidden_units = self.compute_hidden_units(weight_matrix)
            rbm_layer_i = RBM(n_visible_units, n_hidden_units)
            self.rbm_layers.append(rbm_layer_i)
            optimizer = torch.optim.Adam(rbm_layer_i.parameters(), lr=lr, weight_decay=1e-4)
            for epoch in range(num_epochs):
                total_loss = 0.0
                for j in range(0, data.shape[0], batch_size):
                    batch_data = data[j:j + batch_size]
                    if batch_data.shape[1] != n_visible_units:
                        batch_data = batch_data.view(-1, n_visible_units)
                    loss = rbm_layer_i.contrastive_divergence(batch_data, lr)
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                print(f"Epoch {epoch + 1}, Loss: {total_loss / data.shape[0]}")
        self.encode_and_save(original_data)

    def encode_and_save(self, data):
        for rbm_layer in self.rbm_layers:
            rbm_layer.eval()
        encoded_new_data = []
        for i in range(data.shape[0]):
            input_data = data[i, :].unsqueeze(0)
            for rbm_layer in self.rbm_layers:
                if input_data.shape[1] != rbm_layer.W.shape[0]:
                    input_data = input_data.view(1, rbm_layer.W.shape[0])
                input_data = rbm_layer.encode(input_data)
            encoded_data = input_data.squeeze().detach().numpy().flatten()
            encoded_new_data.append(encoded_data)
        encoded_new_data = np.array(encoded_new_data)
        n_features = encoded_new_data.shape[1]
        column_names = [f"DBN{i + 1}" for i in range(n_features)]
        new_data_output = pd.DataFrame(encoded_new_data, columns=column_names)
        new_data_output.to_excel("Result.xlsx", index=False)
        print(f"Encoded output for new data saved")


num_layers = 3
dbn = DBN( num_layers)
dbn.trainy(data, num_epochs=5, batch_size=50, lr=0.01)
