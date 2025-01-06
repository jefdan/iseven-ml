import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
    )
import matplotlib.pyplot as plt
from main import SimpleNN, int_to_binary_array

# Check if a GPU is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model.
model = SimpleNN().to(device)
model_path = "iseven-ml.pth"
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode.
print("Model loaded from disk.")

# Load test data from test_data.csv.
test_data_df = pd.read_csv("test_data.csv")
test_data_list = [
    int_to_binary_array(x, width=16) for x in test_data_df['number']
    ]
test_data = torch.tensor(
    np.array(test_data_list),
    dtype=torch.float32).to(device)

# Testing the model...
predictions = model(test_data)  # Raw outputs (probabilities).
predicted_labels = (predictions > 0.5).float()  # Thresholded to 0 or 1.

# Calculate the accuracy.
accuracy = accuracy_score(
    test_data_df['number'] % 2,
    predicted_labels.cpu().detach().numpy()
    )
print(f"Accuracy: {accuracy:.4f}")

# Generate the confusion matrix.
conf_matrix = confusion_matrix(
    test_data_df['number'] % 2,
    predicted_labels.cpu().detach().numpy()
    )
print("Confusion Matrix:")
print(conf_matrix)

# Generate the classification report.
class_report = classification_report(
    test_data_df['number'] % 2,
    predicted_labels.cpu().detach().numpy(),
    target_names=["even", "odd"]
    )
print("Classification Report:")
print(class_report)

# Calculate the ROC AUC.
roc_auc = roc_auc_score(
    test_data_df['number'] % 2, predictions.cpu().detach().numpy()
    )
print(f"ROC AUC: {roc_auc:.4f}")

# Plot the ROC curve.
fpr, tpr, _ = roc_curve(
    test_data_df['number'] % 2, predictions.cpu().detach().numpy()
    )
plt.figure()
plt.plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label=f'ROC curve (area = {roc_auc:.2f})'
    )
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()
