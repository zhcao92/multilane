import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the MLP neural network model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # MLP architecture
        self.fc1 = nn.Linear(5, 128)  # Input layer (5 features)
        self.fc2 = nn.Linear(128, 256)  # Hidden layer 1
        self.fc3 = nn.Linear(256, 128)  # Hidden layer 2
        self.fc4 = nn.Linear(128, 64)  # Hidden layer 3
        self.fc5 = nn.Linear(64, 2)  # Output layer (2 outputs: follow_accel, follow_dy)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function: ReLU
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # No activation on the output layer
        return x

# Define the ModelTrainer class
class ModelTrainer:
    def __init__(self, data_path, test_size=0.2, lr=0.001, epochs=100, model_save_path="trained_model.pth"):
        self.data_path = data_path
        self.test_size = test_size
        self.lr = lr
        self.epochs = epochs
        self.model_save_path = model_save_path
        
        # Load and process data
        self.load_data()
        
        # Initialize the model, loss function, and optimizer
        self.model = MLP().to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def load_data(self):
        # Load the dataset
        data = pd.read_csv(self.data_path)

        # Input features
        X = data[['lead_x', 'follow_x', 'lead_y', 'lead_speed', 'follow_y', 'follow_speed']].copy()
        X['distance_x'] = X['lead_x'] - X['follow_x']
        X = X[['distance_x', 'lead_y', 'lead_speed', 'follow_y', 'follow_speed']]

        # Output labels
        y = data[['follow_accel', 'follow_dy']]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        # Standardize the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        self.X_train = scaler_X.fit_transform(X_train)
        self.X_test = scaler_X.transform(X_test)

        self.y_train = scaler_y.fit_transform(y_train)
        self.y_test = scaler_y.transform(y_test)

        # Convert data to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(device)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32).to(device)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).to(device)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            # Forward pass
            outputs = self.model(self.X_train)
            loss = self.criterion(outputs, self.y_train)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            loss = self.criterion(outputs, self.y_test)
        print(f'Test Loss: {loss.item():.4f}')

    def save_model(self):
        """Save the trained model to the specified file path."""
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def run(self):
        print("Starting training...")
        self.train()
        print("Starting testing...")
        self.test()

        # Save the model after training and testing
        self.save_model()

# Example usage
if __name__ == "__main__":
    # Initialize the ModelTrainer with the dataset path, learning rate, and epochs
    trainer = ModelTrainer(data_path="lane_change_simulations_data.csv", lr=0.001, epochs=100, model_save_path="imitation_model.pth")

    # Run the training and testing process
    trainer.run()
