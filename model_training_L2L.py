# model_training_L2L.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import os
import argparse

from tqdm import tqdm  # Import tqdm for progress 

from model import TransformerModel, L2LModel, L2LModel_GSP  # Import TransformerModel, L2LModel

# Enable cuDNN benchmark for optimized performance
torch.backends.cudnn.benchmark = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LaneChangeDataset(Dataset):
    def __init__(self, df, max_agents=10):
        """
        Custom Dataset for Lane Change Simulations.

        Parameters:
            df (DataFrame): Preprocessed and scaled DataFrame.
            max_agents (int): Maximum number of surrounding agents.
        """
        self.max_agents = max_agents

        # Create a list of unique samples
        self.samples = []
        for _, row in df.iterrows():
            ego_car_speed = row['ego_car_speed']
            ego_car_x = row['ego_car_x']
            ego_car_y = row['ego_car_y']
            speed_limit = row['speed_limit']

            # Initialize agents as an empty list
            agents = []
            if row['surrounding_agent_exist'] == 1:
                agents.append([row['surrounding_agent_x'], row['surrounding_agent_y'], row['surrounding_agent_speed']])

            self.samples.append({
                'ego_car': [ego_car_x, ego_car_y, ego_car_speed],
                'agents': agents,  # List of [x, y, speed]
                'speed_limit': [speed_limit],
                'target': [row['ego_car_accel'], row['ego_car_dy']]  # Assuming targets are same for grouped agents
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ego_car = sample['ego_car']
        agents = sample['agents']
        speed_limit = sample['speed_limit']
        target = sample['target']

        # Handle surrounding agents
        if len(agents) == 0:
            agents = np.zeros((self.max_agents, 3), dtype=np.float32)  # No agents present
            agent_mask = np.zeros(self.max_agents, dtype=bool)  # No agents to track
        else:
            agents = np.array(agents)
            num_agents = min(len(agents), self.max_agents)
            agents = agents[:num_agents]
            agent_mask = np.ones(num_agents, dtype=bool)
            if num_agents < self.max_agents:
                padding = np.zeros((self.max_agents - num_agents, 3), dtype=np.float32)  # Padding for max_agents
                agents = np.vstack([agents, padding])
                agent_mask = np.concatenate([agent_mask, np.zeros(self.max_agents - num_agents, dtype=bool)])  # Update mask

        # Convert to tensors
        agents = torch.tensor(agents, dtype=torch.float32)
        ego_car = torch.tensor(ego_car, dtype=torch.float32)  # (3,)
        speed_limit = torch.tensor(speed_limit, dtype=torch.float32)  # (1,)
        agent_mask = torch.tensor(agent_mask, dtype=torch.bool)  # (max_agents,)
        target = torch.tensor(target, dtype=torch.float32)  # (2,)

        return {
            'ego_car': ego_car,
            'agents': agents,
            'speed_limit': speed_limit,
            'agent_mask': agent_mask,
            'target': target
        }

# Define the ModelTrainer class
class ModelTrainer:
    def __init__(self, data_path, test_size=0.2, lr=0.0005, epochs=100, 
                 model_save_path="best_model.pth", batch_size=64, 
                 accumulation_steps=1, num_workers=4, patience=10,
                 argumented_data=False, max_agents=10):
        """
        Initializes the ModelTrainer with the given parameters.

        Parameters:
            data_path (str): Path to the CSV data file.
            test_size (float): Proportion of the dataset to include in the test split.
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            model_save_path (str): File path to save the best model.
            batch_size (int): Number of samples per training batch.
            accumulation_steps (int): Number of gradient accumulation steps.
            num_workers (int): Number of subprocesses for data loading.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            speed_limit (float): Speed limit to be included as a feature.
            max_agents (int): Maximum number of surrounding agents.
        """
        self.data_path = data_path
        self.test_size = test_size
        self.lr = lr
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_workers = num_workers
        self.patience = patience
        self.argumented_data = argumented_data
        self.max_agents = max_agents

        # Initialize model, loss, optimizer, scheduler, scaler
        self.model = L2LModel(embed_dim=8, num_heads=2, num_layers=2, dropout=0.0, max_agents=self.max_agents).to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.scaler_amp = GradScaler()

        # Early stopping variables
        self.best_loss = float('inf')
        self.counter = 0

        # Load and prepare data
        self.load_data()

    def generate_augmented_data(self, data):
        """
        Generate augmented data by adjusting x and speed of ego car.
        """

        t_vectors = np.arange(-0.5, 0.6, 0.1)  # -0.5 to 0.5 inclusive
        
        augmented_data = []
        print(f"Generating augmented data with {len(t_vectors)} vectors...")

        for dt in t_vectors:
            # Create a copy of the original data to modify
            modified = data.copy()

            # Adjust ego_car_speed
            modified['ego_car_speed'] = (modified['ego_car_speed'] + dt * modified['ego_car_accel']).clip(lower=0)

            # Adjust ego_car_x and relative surrounding_agent_x
            modified['ego_car_x'] += modified['ego_car_speed'] * dt
            modified['surrounding_agent_x'] -= modified['ego_car_x']
            modified['ego_car_x'] = 0

            # Adjust acceleration if applicable
            modified['ego_car_accel'] -= dt * 0.1

            # Check for extreme values to prevent numerical issues
            relevant_columns = ['ego_car_x', 'ego_car_y', 'ego_car_speed',
                                'surrounding_agent_x', 'surrounding_agent_y', 'surrounding_agent_speed',
                                'speed_limit', 'ego_car_accel', 'ego_car_dy']
            if (modified[relevant_columns].abs() > 1e3).any().any():
                print(f"Skipping vector with dt {dt:.2f} due to extreme values.")
                continue

            augmented_data.append(modified)

        if augmented_data:
            augmented_df = pd.concat(augmented_data, ignore_index=True)
            print(f"Augmented data size: {augmented_df.shape}")
            return augmented_df
        else:
            print("No augmented data generated.")
            return data.copy()

    def check_data_integrity(self, df):
        """Check for NaNs and Infs in the DataFrame."""
        initial_rows = df.shape[0]
        df = df.dropna()
        after_na = df.shape[0]
        if after_na < initial_rows:
            print(f"Removed {initial_rows - after_na} rows containing NaNs.")

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        after_inf = df.shape[0]
        if after_inf < after_na:
            print(f"Removed {after_na - after_inf} rows containing Infs.")

        return df

    def load_data(self):
        """Load, augment, and preprocess the dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The data file {self.data_path} does not exist.")

        print("Loading existing data...")
        data = pd.read_csv(self.data_path)
        print(f"Original data size: {data.shape}")

        # Rename columns to match the new naming convention
        data = data.rename(columns={
            'surrounding_x': 'surrounding_agent_x',
            'surrounding_y': 'surrounding_agent_y',
            'surrounding_speed': 'surrounding_agent_speed',
            'surrounding_agent_exist': 'surrounding_agent_exist',
            'ego_x': 'ego_car_x',
            'ego_y': 'ego_car_y',
            'ego_speed': 'ego_car_speed',
            'ego_desired_speed': 'speed_limit',
            'ego_accel': 'ego_car_accel',
            'ego_dy': 'ego_car_dy'
        })

        # **Set ego_car_x to 0 and adjust surrounding_agent_x accordingly**
        original_ego_x = data['ego_car_x'].copy()
        data['ego_car_x'] = 0  # Fixed ego_x
        data['surrounding_agent_x'] -= original_ego_x  # Adjust surrounding_agent_x
        # data['surrounding_agent_x'] = np.log(data['surrounding_agent_x'] - original_ego_x)  # Log transformation

        # Generate augmented data
        if self.argumented_data:
            augmented_data = self.generate_augmented_data(data)

            # Combine original and augmented data
            combined_data = pd.concat([data, augmented_data], ignore_index=True)
            print(f"Combined data size: {combined_data.shape}")

            # Check and clean data integrity
            combined_data = self.check_data_integrity(combined_data)      
        else:
            combined_data = self.check_data_integrity(data)
        
        print(f"Data size after cleaning: {combined_data.shape}")

        # Feature Engineering: distance_x is already computed during augmentation
        feature_columns = [
            'idx','ego_car_x', 'ego_car_y', 'ego_car_speed', 'surrounding_agent_exist',
            'surrounding_agent_x', 'surrounding_agent_y', 'surrounding_agent_speed',
            'speed_limit'
        ]
        target_columns = ['ego_car_accel', 'ego_car_dy']

        X = combined_data[feature_columns].copy()
        y = combined_data[target_columns]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, shuffle=True
        )
        print(f"Training set size: {X_train.shape}")
        print(f"Testing set size: {X_test.shape}")

        # Create Dataset objects without scaling data
        train_dataset = LaneChangeDataset(X_train.join(y_train), max_agents=self.max_agents)
        test_dataset = LaneChangeDataset(X_test.join(y_test), max_agents=self.max_agents)

        # Create DataLoaders with optimized parameters
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.custom_collate_fn
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.custom_collate_fn
        )
        print("DataLoaders created.")

    def custom_collate_fn(self, batch):
        """
        Custom collate function to handle variable number of surrounding agents.

        Parameters:
            batch (list): List of samples.

        Returns:
            dict: Batched tensors.
        """
        ego_car = torch.stack([item['ego_car'] for item in batch], dim=0)    # (batch_size, 3)
        agents = torch.stack([item['agents'] for item in batch], dim=0)      # (batch_size, max_agents, 3)
        speed_limit = torch.stack([item['speed_limit'] for item in batch], dim=0)  # (batch_size, 1)
        agent_mask = torch.stack([item['agent_mask'] for item in batch], dim=0)    # (batch_size, max_agents)
        target = torch.stack([item['target'] for item in batch], dim=0)  # (batch_size, 2)

        return {
            'ego_car': ego_car,
            'agents': agents,
            'speed_limit': speed_limit,
            'agent_mask': agent_mask,
            'target': target
        }

    def train(self):
        """Train the model with gradient clipping and mixed precision."""
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            # Initialize tqdm progress bar for batches
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                                desc=f"Epoch {epoch}/{self.epochs}", unit="batch")
            for batch_idx, batch in progress_bar:
                # Move data to GPU inside the loop
                ego_car = batch['ego_car'].to(device, non_blocking=True)          # (batch_size, 3)
                agents = batch['agents'].to(device, non_blocking=True)            # (batch_size, max_agents, 3)
                speed_limit = batch['speed_limit'].to(device, non_blocking=True)  # (batch_size, 1)
                agent_mask = batch['agent_mask'].to(device, non_blocking=True)    # (batch_size, max_agents)
                target = batch['target'].to(device, non_blocking=True)            # (batch_size, 2)

                # with autocast():
                outputs, other_output = self.model(ego_car, agents, speed_limit, agent_mask)
                loss = self.criterion(outputs, target) / self.accumulation_steps

                # print ego_car, agents, speed_limit, agent_mask, target, outputs, loss
                # other_output = list(other_output)  # Ensure other_output is a list
                # out1 = other_output[0]
                # out2 = other_output[1]

                # ego_car = ego_car.cpu().numpy()
                # agents = agents.cpu().numpy()
                # speed_limit = speed_limit.cpu().numpy()
                # agent_mask = agent_mask.cpu().numpy()
                # target = target.cpu().numpy()
                # outputs = outputs.detach().cpu().numpy()
                # out1 = out1.detach().cpu().numpy()
                # out2 = out2.detach().cpu().numpy()
                
                # print(ego_car[1], agents[1], speed_limit[1], agent_mask[1], target[1], outputs[1], out1[1], out2[1])
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at Epoch {epoch}, Batch {batch_idx + 1}. Stopping training.")
                    return

                self.optimizer.zero_grad()
                # self.scaler_amp.scale(loss).backward()
                loss.backward()

                # Gradient Clipping
                # self.scaler_amp.unscale_(self.optimizer)
                # clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer Step
                # self.scaler_amp.step(self.optimizer)
                # self.scaler_amp.update()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Update progress bar with current loss
                progress_bar.set_postfix({'loss': f"{loss.item()*self.accumulation_steps:.6f}"})

            # Scheduler Step based on training loss
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.scheduler.step(avg_epoch_loss)

            print(f"Epoch [{epoch}/{self.epochs}], Training Loss: {avg_epoch_loss:.6f}")

            # Early Stopping Check
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.counter = 0
                self.save_checkpoint(epoch, avg_epoch_loss, filename=self.model_save_path)
                print(f"Checkpoint saved at Epoch {epoch} with Loss {avg_epoch_loss:.6f}")
            else:
                self.counter += 1
                print(f"No improvement in loss for {self.counter} epochs.")
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def test(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", unit="batch"):
                # Move data to GPU inside the loop
                ego_car = batch['ego_car'].to(device, non_blocking=True)          # (batch_size, 3)
                agents = batch['agents'].to(device, non_blocking=True)            # (batch_size, max_agents, 3)
                speed_limit = batch['speed_limit'].to(device, non_blocking=True)  # (batch_size, 1)
                agent_mask = batch['agent_mask'].to(device, non_blocking=True)    # (batch_size, max_agents)
                target = batch['target'].to(device, non_blocking=True)            # (batch_size, 2)

                with autocast():
                    # Transformer expects separate inputs: ego, agents, speed_limit, agent_mask
                    outputs, _ = self.model(ego_car, agents, speed_limit, agent_mask)
                    loss = self.criterion(outputs, target)
                total_loss += loss.item()

                # # print the value of loss.item()
                # print(f"Loss: {loss.item()}")

                # # Combine the ego_car, outputs, and targets
                # combined = torch.cat((ego_car[:, 0:3], outputs[:, 0:2], target[:, 0:2]), dim=1).cpu().numpy()

                # # Print the rounded tensor
                # print(f"Combined: {np.round(combined, 2)}")

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_test_loss = total_loss / len(self.test_loader)
        print(f"Test Loss (Scaled): {avg_test_loss:.6f}")

    def save_checkpoint(self, epoch, loss, filename="checkpoint.pth"):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'scaler_state_dict': self.scaler_amp.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="checkpoint.pth"):
        """Load model checkpoint."""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler_amp.load_state_dict(checkpoint['scaler_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            self.best_loss = loss
            self.counter = 0  # Reset early stopping counter
            print(f"Checkpoint loaded from {filename} at Epoch {epoch} with Loss {loss:.6f}")
            return epoch, loss
        else:
            print(f"No checkpoint found at {filename}")
            return None, None

    def run(self):
        """Run the entire training and testing pipeline."""
        print("Starting Training...")
        self.train()
        print("Training Completed.")

        print("Starting Testing...")
        self.test()
        print("Testing Completed.")

        print("Saving the best model...")
        # The best model is already saved during training
        # Optionally, load it for further use
        # self.load_checkpoint(self.model_save_path)

# Main execution function
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Neural Network Model for Ego Car Control")
    parser.add_argument('--data_file', type=str, default="lane_change_simulations_data.csv",
                        help='Path to the CSV data file.')
    parser.add_argument('--test_size', type=float, default=0.01,
                        help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs.')
    parser.add_argument('--model_save_path', type=str, default="best_L2L_model.pth",
                        help='File path to save the best model.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of samples per training batch.')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of subprocesses for data loading.')
    parser.add_argument('--patience', type=int, default=100,
                        help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--argumented_data', type=bool, default=False,
                        help='Augment data by adjusting x and speed of ego car.')
    parser.add_argument('--speed_limit', type=float, default=60.0,
                        help='Speed limit to be included as a feature.')
    parser.add_argument('--max_agents', type=int, default=10,
                        help='Maximum number of surrounding agents to handle.')

    args = parser.parse_args()

    # Check if the data file exists
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"The data file {args.data_file} does not exist. Please ensure it is present in the directory.")

    # Initialize the trainer with appropriate parameters
    trainer = ModelTrainer(
        data_path=args.data_file,
        test_size=args.test_size,
        lr=args.lr,  # Lowered learning rate to prevent divergence
        epochs=args.epochs,
        model_save_path=args.model_save_path,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        num_workers=args.num_workers,  # Adjust based on CPU cores
        patience=args.patience,  # Early stopping patience
        argumented_data=args.argumented_data,
        max_agents=args.max_agents
    )

    # Run the training and testing process
    trainer.run()

if __name__ == "__main__":
    main()
