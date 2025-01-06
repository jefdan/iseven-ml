import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Check if a GPU is available.
# GPUs are very fast and cool. If you have one, you should use it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate the 'dataset'.
# Integers from 1 to 65535.
# The maximum is 65535 because we are using a 16-bit binary representation.
data = np.arange(1, 65536)
# 1 for odd, 0 for even.
labels = np.array([1 if x % 2 != 0 else 0 for x in data])


# Convert the 'data' to binary representation.
# If the data isn't converted to binary, the model won't be able
# to learn a pattern because there isn't really one.
def int_to_binary_array(x, width=16):
    return np.array(list(np.binary_repr(x, width=width)), dtype=np.float32)


def binary_array_to_int(binary_array):
    return int("".join(binary_array.astype(int).astype(str)), 2)


def labels_to_human_readable(predicted_labels):
    return [
        "odd" if label == 1
        else "even" for label in predicted_labels.cpu().numpy()
    ]


binary_data = np.array([int_to_binary_array(x) for x in data])

# Convert to PyTorch tensors and move to device.
data = torch.tensor(binary_data, dtype=torch.float32).to(device)
labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


# Define the model.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


model = SimpleNN().to(device)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


model.apply(weights_init)

# Define loss and optimiser.
criterion = nn.BCELoss()  # Binary Cross Entropy Loss.
optimiser = optim.SGD(model.parameters(), lr=0.001)

# Check the dataset's balance.
odd_count = sum(labels.cpu().numpy())
even_count = len(labels) - odd_count
print(f"Odd count: {odd_count}, Even count: {even_count}")

# Train the model!
num_epochs = 25000  # Number of epochs, modify this!
model_path = "iseven-ml.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from disk.")
else:
    for epoch in range(num_epochs):
        model.train()

        # Forward pass.
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass and optimisation.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Print the progress, every 10 epochs along
        # with the loss value.
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), model_path)
    print("Model saved to disk.")

# Load the test data from test_data.csv.
test_data_df = pd.read_csv("test_data.csv")
test_data_list = [
    int_to_binary_array(
        x,
        width=16) for x in test_data_df['number']]
test_data = torch.tensor(
    np.array(test_data_list),
    dtype=torch.float32).to(device)

# Test the model!
model.eval()
predictions = model(test_data)
# If the probability is greater than 0.5, the number is odd.
# Otherwise, it is even.
predicted_labels = (predictions > 0.5).float()

# Format the output as a table and save to output_predictions.csv.
test_data_int = test_data_df['number'].tolist()
predictions_list = predictions.cpu().view(-1).tolist()
human_readable_labels = labels_to_human_readable(predicted_labels)

output_df = pd.DataFrame({
    'Test Data': test_data_int,
    'Prediction': [f"{pred:.4f}" for pred in predictions_list],
    'Label': human_readable_labels
})

output_df.to_csv("output_predictions.csv", index=False)

# Finally print the small test that we have done here to the
# console.
print(f"{'Test Data':<10} {'Prediction':<12} {'Label':<6}")
print("-" * 30)
for i in range(len(test_data_int)):
    print(
        f"{test_data_int[i]:<10} "
        f"{predictions_list[i]:<12.4f} "
        f"{human_readable_labels[i]:<6}"
    )
