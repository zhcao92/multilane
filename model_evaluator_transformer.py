# model_evaluator_L2L.py

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import cv2
from tqdm import tqdm
import logging
import argparse

from model import TransformerModel  # Import the TransformerModel class

# Constants for vehicle colors
LEAD_CAR_COLOR = '#F8B37F'
FOLLOW_CAR_COLOR = '#202F39'

def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("model_evaluator.log"),
            logging.StreamHandler()
        ]
    )

class ModelEvaluator:
    def __init__(self, model_path, data_path, device, ego_desired_speed=60/3.6, num_agents=1, scenario_time=60, time_step=0.1):
        """
        Initializes the ModelEvaluator with paths to the model, dataset, and simulation parameters.

        Args:
            model_path (str): Path to the trained model checkpoint.
            data_path (str): Path to the dataset CSV file.
            device (torch.device): Device to run the model on.
            num_agents (int): Number of surrounding agents to include (0 to 10).
            scenario_time (float): Total time for the simulation in seconds.
            time_step (float): Time step for the simulation in seconds.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.num_agents = min(num_agents, 10)  # Ensure num_agents does not exceed max_agents=10
        self.scenario_time = scenario_time
        self.time_step = time_step
        self.device = device
        self.ego_desired_speed = ego_desired_speed

        # Initialize the model
        self.model = self.load_trained_model()

        # Initialize lists to store simulation data
        self.initialize_simulation_data()

    def load_trained_model(self):
        """
        Loads the trained Transformer model from the specified checkpoint.

        Returns:
            nn.Module: The loaded Transformer model set to evaluation mode.
        """
        try:
            max_agents = 10  # As per instruction, set max_agents=10
            model = TransformerModel(max_agents=max_agents).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"Loaded Transformer model state_dict from {self.model_path}")
            else:
                model.load_state_dict(checkpoint)
                logging.info(f"Loaded Transformer model directly from {self.model_path}")

            model.eval()
            return model

        except FileNotFoundError:
            logging.error(f"Model file not found at {self.model_path}")
            raise
        except KeyError as e:
            logging.error(f"Key error when loading model: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading model: {e}")
            raise

    def initialize_simulation_data(self):
        """
        Initializes lists to store simulation data for visualization.
        """
        self.time_points = []
        # Initialize data storage for surrounding agents
        self.surrounding_agents_x_points = [[] for _ in range(10)]  # Up to 10 agents
        self.surrounding_agents_y_points = [[] for _ in range(10)]
        self.surrounding_agents_speed_points = [[] for _ in range(10)]

        # Initialize data storage for ego car
        self.ego_car_x_points = []
        self.ego_car_y_points = []
        self.ego_car_speed_points = []
        self.ego_car_accel_points = []
        self.ego_car_dy_points = []

    def evaluate_multilane_scenario(self):
        """
        Evaluates the trained Transformer model in a multilane driving scenario simulation.
        """
        # Initialize vehicle states for surrounding agents
        surrounding_agents = []
        for i in range(self.num_agents):
            agent = {
                'x': 50.0 + i*10,    # Start from 50m and space by 10m
                'y': 0.0,            # Lane position (0: right lane)
                'speed': 40.0 / 3.6, # Constant speed of 30 km/h converted to m/s
            }
            surrounding_agents.append(agent)

        # Initialize ego car state
        ego_car = {
            'x': 0.0,
            'y': 0.0,
            'speed': 0.0,  # Starts from 0 km/h
            'accel': 0.0,
            'dy': 0.0,
        }

        # Define speed limit (e.g., 30 km/h)
        speed_limit = self.ego_desired_speed

        num_steps = int(self.scenario_time / self.time_step)
        logging.info(f"Starting simulation for {self.scenario_time} seconds with time step {self.time_step}s")
        logging.info(f"Number of Surrounding Agents: {self.num_agents}")

        with torch.no_grad():
            for step in tqdm(range(num_steps), desc="Simulating Scenario"):
                current_time = step * self.time_step

                # Prepare inputs for TransformerModel
                ego = [0, ego_car['y'], ego_car['speed']] # Ego car x is always 0

                agents = []
                agent_mask = []
                for agent in surrounding_agents:
                    agent_x_rel = agent['x'] - ego_car['x']
                    agent_y = agent['y']
                    agent_speed = agent['speed']
                    agents.append([agent_x_rel, agent_y, agent_speed])
                    agent_mask.append(1)  # Active agent

                # Pad agents list to have exactly 10 agents
                while len(agents) < 10:
                    agents.append([0.0, 0.0, 0.0])  # Padding with zeros
                    agent_mask.append(0)             # Inactive agent

                # Convert inputs to tensors
                ego_tensor = torch.tensor([ego], dtype=torch.float32).to(self.device)       # Shape: (1, 3)
                agents_tensor = torch.tensor([agents], dtype=torch.float32).to(self.device) # Shape: (1, 10, 3)
                agent_mask_tensor = torch.tensor([agent_mask], dtype=torch.bool).to(self.device) # Shape: (1, 10)

                speed_limit_tensor = torch.tensor([[speed_limit]], dtype=torch.float32).to(self.device)

                # Predict acceleration and lane change rate using the Transformer model
                outputs = self.model(ego_tensor, agents_tensor, speed_limit_tensor, agent_mask_tensor)
                outputs_np = outputs.cpu().numpy()

                # Check for non-finite values in outputs
                if not np.all(np.isfinite(outputs_np)):
                    logging.error(f"Model output contains non-finite values: {outputs_np}")
                    logging.debug(f"Current ego: {ego}, agents: {agents}, speed_limit: {speed_limit}, agent_mask: {agent_mask}")
                    accel, dy = 0.0, 0.0  # Assign default safe values
                else:
                    accel, dy = outputs_np[0]

                    # Clip accel to [-2, 2] m/s²
                    if accel < -2:
                        logging.warning(f"Clipping accel from {accel:.2f} to -2.00")
                        accel = -2.0
                    elif accel > 2:
                        logging.warning(f"Clipping accel from {accel:.2f} to 2.00")
                        accel = 2.0

                    # Clip dy to [-0.25, 0.25] (lane change rate)
                    if dy < -0.25:
                        logging.warning(f"Clipping dy from {dy:.2f} to -0.25")
                        dy = -0.25
                    elif dy > 0.25:
                        logging.warning(f"Clipping dy from {dy:.2f} to 0.25")
                        dy = 0.25

                # Update ego car state
                ego_car['accel'] = accel
                ego_car['dy'] = dy  # Integrate lane change rate

                # Update ego car speed and position
                ego_car['speed'] += ego_car['accel'] * self.time_step
                ego_car['speed'] = max(0, ego_car['speed'])  # Ensure speed is non-negative
                ego_car['x'] += ego_car['speed'] * self.time_step
                ego_car['y'] += ego_car['dy'] * self.time_step

                # Clip ego_car['y'] to [-0.5, 1.5] to stay within lanes
                if ego_car['y'] < -0.5:
                    logging.warning(f"Clipping ego_car y-position from {ego_car['y']:.2f} to -0.5")
                    ego_car['y'] = -0.5
                elif ego_car['y'] > 1.5:
                    logging.warning(f"Clipping ego_car y-position from {ego_car['y']:.2f} to 1.5")
                    ego_car['y'] = 1.5

                # Update surrounding agents' positions (constant speed)
                for i, agent in enumerate(surrounding_agents):
                    agent['x'] += agent['speed'] * self.time_step

                # Store data for visualization
                self.time_points.append(current_time)
                for i in range(10):
                    if i < self.num_agents:
                        self.surrounding_agents_x_points[i].append(surrounding_agents[i]['x'])
                        self.surrounding_agents_y_points[i].append(surrounding_agents[i]['y'])
                        self.surrounding_agents_speed_points[i].append(surrounding_agents[i]['speed'] * 3.6)  # Convert to km/h
                    else:
                        # Append None or NaN for inactive agents
                        self.surrounding_agents_x_points[i].append(np.nan)
                        self.surrounding_agents_y_points[i].append(np.nan)
                        self.surrounding_agents_speed_points[i].append(np.nan)

                self.ego_car_x_points.append(ego_car['x'])
                self.ego_car_y_points.append(ego_car['y'])
                self.ego_car_speed_points.append(ego_car['speed'] * 3.6)  # Convert to km/h
                self.ego_car_accel_points.append(ego_car['accel'])
                self.ego_car_dy_points.append(ego_car['dy'])

    def plot_results(self, save_path="evaluation_multilane_scenario.png"):
        """
        Plots the results of the simulation.

        Args:
            save_path (str): Path to save the generated plot.
        """
        num_plots = 5
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))

        # Plot 1: Surrounding agents and ego car X positions vs time
        if self.num_agents > 0:
            for i in range(self.num_agents):
                relative_x_positions = [agent_x - ego_x for agent_x, ego_x in zip(self.surrounding_agents_x_points[i], self.ego_car_x_points)]
                axs[0].plot(self.time_points, relative_x_positions, label=f"Agent {i+1} relative X Position")
        else:
            axs[0].plot(self.time_points, self.ego_car_x_points, label="Ego Car X Position", color='black', linewidth=2)
        axs[0].set_title("Surrounding Agents and Ego Car X Positions vs Time")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("X Position [m]")
        axs[0].legend()
        axs[0].grid(False)

        # Plot 2: Surrounding agents and ego car Y (lane) positions vs time
        for i in range(self.num_agents):
            axs[1].plot(self.time_points, self.surrounding_agents_y_points[i], label=f"Agent {i+1} Lane Position (Y)")
        axs[1].plot(self.time_points, self.ego_car_y_points, label="Ego Car Lane Position (Y)", color='black', linewidth=2)
        axs[1].set_title("Surrounding Agents and Ego Car Lane Positions (Y) vs Time")
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Y (Lane Position)")
        axs[1].set_ylim(-0.5, 1.5)
        axs[1].legend()
        axs[1].grid(False)

        # Plot 3: Surrounding agents and ego car speeds vs time
        for i in range(self.num_agents):
            axs[2].plot(self.time_points, self.surrounding_agents_speed_points[i], label=f"Agent {i+1} Speed (km/h)")
        axs[2].plot(self.time_points, self.ego_car_speed_points, label="Ego Car Speed (km/h)", color='black', linewidth=2)
        axs[2].set_title("Surrounding Agents and Ego Car Speeds vs Time")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Speed [km/h]")
        axs[2].legend()
        axs[2].grid(False)

        # Plot 4: Ego car acceleration vs time
        axs[3].plot(self.time_points, self.ego_car_accel_points, label="Ego Car Acceleration (m/s²)", color='green')
        axs[3].set_title("Ego Car Acceleration vs Time")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel("Acceleration [m/s²]")
        axs[3].legend()
        axs[3].grid(False)

        # Plot 5: Ego car lane change (dy) vs time
        axs[4].plot(self.time_points, self.ego_car_dy_points, label="Ego Car Lane Change (dy)", color='red')
        axs[4].set_title("Ego Car Lane Change (dy) vs Time")
        axs[4].set_xlabel("Time [s]")
        axs[4].set_ylabel("Lane Change (dy)")
        axs[4].set_ylim(-0.25, 0.25)
        axs[4].legend()
        axs[4].grid(False)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Simulation results plotted and saved to {save_path}")

    def generate_video_images(self, evaluator, output_dir="frames", include_agent=True):
        """
        Generates individual frames for the simulation to create a video.

        Args:
            evaluator (ModelEvaluator): Instance of the ModelEvaluator class.
            output_dir (str): Directory to save the frame images.
            include_agent (bool): Whether to include surrounding agents in the video.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(20, 4))

        with tqdm(total=len(evaluator.time_points), desc="Generating Frames", unit="frame") as pbar:
            for frame in range(len(evaluator.time_points)):
                ax.clear()

                # Set plot limits and aesthetics
                ax.set_xlim(-10, 80)  # Fixed range for x-axis
                ax.set_ylim(-1, 2)    # Adjusted y-axis to accommodate lane clipping
                ax.set_yticks([])      # Remove the y-axis numbers

                # Set up the lanes again after clearing
                ax.axhline(y=-0.5, color='black', linestyle='-')
                ax.axhline(y=0.5, color='black', linestyle='-')
                ax.axhline(y=1.5, color='black', linestyle='-')

                ax.axhline(y=0.0, color='grey', linestyle='--')
                ax.axhline(y=1.0, color='grey', linestyle='--')

                # Plot surrounding agents
                if include_agent and evaluator.num_agents > 0:
                    for i in range(evaluator.num_agents):
                        agent_x = evaluator.surrounding_agents_x_points[i][frame] - evaluator.ego_car_x_points[frame]
                        agent_y = evaluator.surrounding_agents_y_points[i][frame]
                        agent_rect = plt.Rectangle((agent_x - 2.5, agent_y - 0.3), 5, 0.6, color=LEAD_CAR_COLOR)
                        ax.add_patch(agent_rect)
                        ax.text(agent_x, agent_y + 0.5, f"Agent {i+1}", ha='center', va='bottom', fontsize=8, color='black')

                        # Annotate agent speed
                        ax.text(agent_x, agent_y - 0.8, f"Speed: {evaluator.surrounding_agents_speed_points[i][frame]:.1f} km/h",
                                ha='center', va='top', fontsize=8, color='black')

                # Plot ego car
                ego_car_x = 0
                ego_car_y = evaluator.ego_car_y_points[frame]
                ego_car_rect = plt.Rectangle((ego_car_x - 2.5, ego_car_y - 0.3), 5, 0.6, color=FOLLOW_CAR_COLOR)
                ax.add_patch(ego_car_rect)
                ax.text(ego_car_x, ego_car_y + 0.5, "Ego Car", ha='center', va='bottom', fontsize=8, color='black')

                # Annotate ego car speed
                ax.text(ego_car_x, ego_car_y - 0.8, f"Speed: {evaluator.ego_car_speed_points[frame]:.1f} km/h",
                        ha='center', va='top', fontsize=8, color='black')

                # Save frame
                frame_filename = os.path.join(output_dir, f"frame_{frame:04d}.png")
                plt.savefig(frame_filename)
                pbar.update(1)

        plt.close()
        logging.info(f"Frame images generated in directory: {output_dir}")

    def create_video_from_images(self, image_dir, video_file, fps=30, remove_images=False):
        """
        Creates a video from a sequence of saved images.

        Args:
            image_dir (str): Directory containing the frame images.
            video_file (str): Output video file name.
            fps (int): Frames per second for the video.
            remove_images (bool): Whether to remove individual frame images after creating the video.
        """
        frame_files = sorted([
            os.path.join(image_dir, img) 
            for img in os.listdir(image_dir) 
            if img.endswith(".png")
        ])

        if not frame_files:
            logging.error(f"No frame images found in {image_dir}.")
            return

        # Read the first frame to get video dimensions
        frame = cv2.imread(frame_files[0])
        if frame is None:
            logging.error(f"Failed to read the first frame from {frame_files[0]}.")
            return
        height, width, layers = frame.shape

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

        for frame_file in tqdm(frame_files, desc="Creating Video"):
            frame = cv2.imread(frame_file)
            if frame is None:
                logging.warning(f"Failed to read frame: {frame_file}. Skipping.")
                continue
            video_writer.write(frame)

        video_writer.release()
        logging.info(f"Video created and saved as {video_file}")

        if remove_images:
            for frame_file in frame_files:
                os.remove(frame_file)
            os.rmdir(image_dir)
            logging.info(f"Removed frame images from {image_dir}")

# Define a function to parse command-line arguments
def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate Neural Network Model for Ego Car Control")
    parser.add_argument('--data_file', type=str, default="lane_change_simulations_data.csv",
                        help='Path to the CSV data file.')
    parser.add_argument('--model_path', type=str, default="best_L2L_model.pth",
                        help='Path to the trained model file.')
    parser.add_argument('--ego_desired_speed', type=float, default=60.0,
                        help='Speed limit to be included as a feature. (km/h)')
    parser.add_argument('--num_agents', type=int, default=1,
                        help='Number of surrounding agents in simulator (TBD).')
    parser.add_argument('--generate_frame_pics', type=bool, default=False,
                        help='Generate frame images for video creation.')
    parser.add_argument('--generate_video', type=bool, default=False,
                        help='Generate video from frame images.')

    return parser.parse_args()

def main():
    """
    Main execution function for the model evaluator.
    """

    args = parse_args()

    # Check if the data file exists
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"The data file {args.data_file} does not exist. Please ensure it is present in the directory.")
    
    # Check if the model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"The model file {args.model_path} does not exist. Please ensure it is present in the directory.")
    
    setup_logging()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")


    # Initialize the ModelEvaluator
    evaluator = ModelEvaluator(
        data_path=args.data_file,
        model_path=args.model_path,
        device=device,
        ego_desired_speed=args.ego_desired_speed/3.6, # in m/s
        num_agents=args.num_agents,
        scenario_time=120,  # 120 seconds simulation
        time_step=0.1       # 0.1 second time step
    )

    # Run the simulation
    evaluator.evaluate_multilane_scenario()

    # Plot the results
    evaluator.plot_results(save_path="evaluation_multilane_scenario.png")

    # Generate video frames (TBD)
    # Set include_agent=True to visualize surrounding agents in the video
    # generate_video_images is defined outside the class for flexibility
    if args.generate_frame_pics:
        evaluator.generate_video_images(evaluator, output_dir="frames", include_agent=True)

    # Create a video from the frames
    if args.generate_video:
        evaluator.create_video_from_images(
            image_dir="frames",
            video_file="multilane_driving_simulation.mp4",
            fps=30,
            remove_images=False  # Set to True to delete frames after video creation
        )

if __name__ == "__main__":
    main()
