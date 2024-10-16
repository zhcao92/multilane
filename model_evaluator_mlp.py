import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
import cv2
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEAD_CAR_COLOR = '#F8B37F'
FOLLOW_CAR_COLOR = '#202F39'

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(5, 128)  # Input layer (5 features)
        self.fc2 = nn.Linear(128, 256)  # Hidden layer 1
        self.fc3 = nn.Linear(256, 128)  # Hidden layer 2
        self.fc4 = nn.Linear(128, 64)  # Hidden layer 3
        self.fc5 = nn.Linear(64, 2)  # Output layer (2 outputs: follow_accel, follow_dy)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class ModelEvaluator:
    def __init__(self, model_path, data_path, scenario_time=60, time_step=0.1):
        self.model_path = model_path
        self.data_path = data_path
        self.scenario_time = scenario_time
        self.time_step = time_step

        # Load the trained model
        self.model = self.load_trained_model()

        # Load the dataset and fit the scalers
        self.scaler_X, self.scaler_y, self.feature_names = self.load_and_fit_scalers()

        # Store time points and vehicle states for visualization
        self.time_points = []
        self.lead_x_points = []
        self.lead_y_points = []
        self.lead_speed_points = []

        self.follow_x_points = []
        self.follow_y_points = []
        self.follow_speed_points = []
        self.follow_accel_points = []
        self.follow_dy_points = []
        
    def load_trained_model(self):
        """Load the trained model from file."""
        model = MLP().to(device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {self.model_path}")
        return model

    def load_and_fit_scalers(self):
        """Load the dataset and fit the scalers based on it."""
        data = pd.read_csv(self.data_path)

        # Input features
        X = data[['lead_x', 'follow_x', 'lead_y', 'lead_speed', 'follow_y', 'follow_speed']].copy()
        X['distance_x'] = X['lead_x'] - X['follow_x']
        X = X[['distance_x', 'lead_y', 'lead_speed', 'follow_y', 'follow_speed']]

        # Output labels
        y = data[['follow_accel', 'follow_dy']]

        # Initialize and fit scalers
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # Store the feature names
        feature_names = X.columns

        scaler_X.fit(X)
        scaler_y.fit(y)

        print("Scalers fitted on the dataset")
        return scaler_X, scaler_y, feature_names

    def evaluate_multilane_scenario(self):
        """Evaluate the trained model in a multilane driving scenario."""
        # Initialize vehicle states for the evaluation scenario
        lead_vehicle = {
            'x': 50,  # starting x position
            'y': 0,  # lane position (right lane)
            'speed': 30 / 3.6,  # constant speed of 30 km/h (8.33 m/s)
        }
        
        follow_vehicle = {
            'x': 0,
            'y': 0,
            'speed': 0,  # starts from 0 km/h
            'accel': 0,
            'dy': 0,
        }

        # Run the scenario
        for t in range(int(self.scenario_time / self.time_step)):
            time = t * self.time_step

            # Prepare the input array
            input_array = [[lead_vehicle['x'] - follow_vehicle['x'], lead_vehicle['y'], lead_vehicle['speed'], follow_vehicle['y'], follow_vehicle['speed']]]

            # Convert input array to DataFrame with correct feature names
            input_df = pd.DataFrame(input_array, columns=self.feature_names)

            # Normalize the input using the scaler
            inputs = torch.tensor(self.scaler_X.transform(input_df), dtype=torch.float32).to(device)

            # Predict the follow vehicle's acceleration and lane change
            with torch.no_grad():
                outputs = self.model(inputs)

            # Inverse transform the output using the scaler to get actual values
            outputs = self.scaler_y.inverse_transform(outputs.cpu().numpy())

            # Update follow vehicle's states
            follow_vehicle['accel'] = outputs[0][0]  # acceleration
            follow_vehicle['dy'] = outputs[0][1]  # lane change

            # Update lead vehicle position (constant speed)
            lead_vehicle['x'] += lead_vehicle['speed'] * self.time_step

            # Update speed and position of the follow vehicle
            follow_vehicle['speed'] += follow_vehicle['accel'] * self.time_step
            follow_vehicle['x'] += follow_vehicle['speed'] * self.time_step
            follow_vehicle['y'] += follow_vehicle['dy'] * self.time_step

            # Store the data for plotting
            self.time_points.append(time)
            self.follow_x_points.append(follow_vehicle['x'])
            self.follow_y_points.append(follow_vehicle['y'])
            self.follow_speed_points.append(follow_vehicle['speed'] * 3.6)  # convert to km/h
            self.follow_accel_points.append(follow_vehicle['accel'])
            self.follow_dy_points.append(follow_vehicle['dy'])
            self.lead_x_points.append(lead_vehicle['x'])
            self.lead_y_points.append(lead_vehicle['y'])
            self.lead_speed_points.append(lead_vehicle['speed'] * 3.6) # convert to km/h


    def plot_results(self):
        """Plot the results of the scenario evaluation."""
        fig, axs = plt.subplots(5, 1, figsize=(10, 20))

        # Plot 1: Lead and follow vehicle X positions vs time
        axs[0].plot(self.time_points, self.lead_x_points, label="Lead Vehicle X Position")
        axs[0].plot(self.time_points, self.follow_x_points, label="Follow Vehicle X Position")
        axs[0].set_title("Lead and Follow Vehicle X Positions vs Time")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("X Position [m]")
        axs[0].legend()

        # Plot 2: Lead and follow vehicle Y (lane) positions vs time
        axs[1].plot(self.time_points, self.lead_y_points, label="Lead Vehicle Lane Position (Y)")
        axs[1].plot(self.time_points, self.follow_y_points, label="Follow Vehicle Lane Position (Y)")
        axs[1].set_title("Lead and Follow Vehicle Lane Positions (Y) vs Time")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Y (Lane Position)")
        axs[1].legend()

        # Plot 3: Lead and follow vehicle speeds vs time
        axs[2].plot(self.time_points, self.lead_speed_points, label="Lead Vehicle Speed (km/h)")
        axs[2].plot(self.time_points, self.follow_speed_points, label="Follow Vehicle Speed (km/h)")
        axs[2].set_title("Lead and Follow Vehicle Speeds vs Time")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Speed [km/h]")
        axs[2].legend()

        # Plot 4: Follow vehicle acceleration vs time
        axs[3].plot(self.time_points, self.follow_accel_points, label="Follow Vehicle Acceleration (m/s²)")
        axs[3].set_title("Follow Vehicle Acceleration vs Time")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel("Acceleration [m/s²]")
        axs[3].legend()

        # Plot 5: Follow vehicle lane change (dy) vs time
        axs[4].plot(self.time_points, self.follow_dy_points, label="Follow Vehicle Lane Change (dy)")
        axs[4].set_title("Follow Vehicle Lane Change (dy) vs Time")
        axs[4].set_xlabel("Time [s]")
        axs[4].set_ylabel("Lane Change (dy)")
        axs[4].legend()

        plt.tight_layout()
        plt.savefig("evaluation_multilane_scenario_updated.png")
        plt.show()

    def generate_video_images(self, output_dir="frames"):
        """Generate a video showing the driving process by saving individual frames and compiling them."""
        
        # Create a directory to store the individual frames
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig, ax = plt.subplots(figsize=(20, 4))


        with tqdm(total=len(self.time_points), desc="Generating Frames", unit="frame") as pbar:

            # Iterate through each time step to generate frames
            for frame in range(len(self.time_points)):
                ax.clear()  # Clear previous plot

                ax.set_xlim(-10, 80)  # Fixed range for x-axis
                ax.set_yticks([])  # Remove the y-axis numbers

                # Set up the lanes again after clearing
                ax.axhline(y=-0.5, color='black', linestyle='-')
                ax.axhline(y=0.5, color='black', linestyle='-')
                ax.axhline(y=1.5, color='black', linestyle='-')

                ax.axhline(y=0.0, color='grey', linestyle='--')
                ax.axhline(y=1.0, color='grey', linestyle='--')

                follow_x = 0
                lead_x = self.lead_x_points[frame] - self.follow_x_points[frame]

                # Add rectangles representing the vehicles
                lead_vehicle_plot = plt.Rectangle((lead_x, self.lead_y_points[frame]-0.25), 5, 0.5, color=LEAD_CAR_COLOR, label='Lead Vehicle')
                follow_vehicle_plot = plt.Rectangle((follow_x, self.follow_y_points[frame]-0.25), 5, 0.5, color=FOLLOW_CAR_COLOR, label='Follow Vehicle')

                ax.add_patch(lead_vehicle_plot)
                ax.add_patch(follow_vehicle_plot)

                # Add speed vectors as arrows (scaled for visualization)
                lead_speed = self.lead_speed_points[frame]
                follow_speed = self.follow_speed_points[frame]

                lead_vector = ax.arrow(lead_x+2.5, self.lead_y_points[frame], lead_speed*0.2, 0, head_width=0.1, head_length=1, fc=LEAD_CAR_COLOR, ec=LEAD_CAR_COLOR)
                follow_vector = ax.arrow(follow_x+2.5, self.follow_y_points[frame], follow_speed*0.2, 0, head_width=0.1, head_length=1, fc=FOLLOW_CAR_COLOR, ec=FOLLOW_CAR_COLOR)

                # Display speed values on the speed vectors
                ax.text(lead_x+2.5 + lead_speed*0.1 + 4, self.lead_y_points[frame]-0.125, f"{lead_speed:.1f} km/h", color=LEAD_CAR_COLOR, fontsize=16, verticalalignment='center')
                ax.text(follow_x+2.5 + follow_speed*0.1 + 4, self.follow_y_points[frame]-0.125, f"{follow_speed:.1f} km/h", color=FOLLOW_CAR_COLOR, fontsize=16, verticalalignment='center')

                # Set the legend and plot title
                ax.legend(loc='upper right')

                # Save the frame as an image file
                frame_filename = os.path.join(output_dir, f"frame_{frame:04d}.png")
                plt.savefig(frame_filename)

                # Update the progress bar
                pbar.update(1)

    def create_video_from_images(self, image_dir, video_file, fps=30, remove_images=False):
        """Create a video from a sequence of saved images."""

        frame_files = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".png")])

        if len(frame_files) == 0:
            print(f"No frames found in {image_dir}")
            return

        # Read the first image to get frame size
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape

        # Initialize video writer
        video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Write each frame to the video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        print(f"Video saved as {video_file}")

        # Optionally, clean up the image directory by removing all frame files
        if remove_images:
            for filename in frame_files:
                os.remove(filename)
            os.rmdir(image_dir)

if __name__ == "__main__":
    # Initialize the ModelEvaluator with the trained model path and dataset path
    evaluator = ModelEvaluator(
        model_path="imitation_model.pth",
        data_path="lane_change_simulations_data.csv",
        scenario_time=120,  # 60 seconds simulation
        time_step=0.1  # 0.1 second time step
    )

    # Evaluate the multilane driving scenario
    evaluator.evaluate_multilane_scenario()

    # Plot the results
    evaluator.plot_results()

    # Generate the video
    evaluator.generate_video_images(output_dir = "frames")

    # Create a video from the saved frames
    evaluator.create_video_from_images(image_dir = "frames", video_file = "multilane_driving.mp4", remove_images = False)
