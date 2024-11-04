import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants for simulation
TIME_STEP = 0.1  # Time step in seconds
SIMULATION_TIME = 60  # Total simulation time in seconds

# Parameters for IDM model
T = 1.5  # Safe time headway in seconds
a = 1.0  # Maximum acceleration in m/s^2
b = 3.0  # Comfortable deceleration in m/s^2
s0 = 2.0  # Minimum distance to lead vehicle in meters
delta = 4.0  # Acceleration exponent

# MOBIL parameters
p = 0.5  # Politeness factor
lane_change_threshold = 0.2  # Threshold for lane change

# Vehicle class to represent a single vehicle in the simulation
class Vehicle:
    def __init__(self, lane, position, velocity, desired_speed, accel=0, dy=0, lane_change=False):
        self.lane = lane
        self.position = position
        self.velocity = velocity
        self.desired_speed = desired_speed  # Desired speed is now a parameter
        self.accel = accel
        self.dy = dy

        self.last_lane = lane
        self.last_position = position
        self.last_velocity = velocity

        # Lane change state variables
        self.lane_change = lane_change
        self.lane_change_start = False
        self.lane_change_duration = 4  # Lane change duration in seconds

    def update(self):
        # Update the last position and velocity
        self.last_position, self.last_velocity, self.last_lane = self.position, self.velocity, self.lane

        # Update the position and velocity using simple kinematics
        self.velocity += self.accel * TIME_STEP
        self.position += self.velocity * TIME_STEP
        self.lane += self.dy

    def idm_acceleration(self, lead_vehicle):
        """Compute the IDM acceleration for car-following."""
        if lead_vehicle is not None:
            delta_v = self.velocity - lead_vehicle.velocity
            s_star = s0 + self.velocity * T + (self.velocity * delta_v) / (2 * np.sqrt(a * b))
            s = lead_vehicle.position - self.position
            return a * (1 - (self.velocity / self.desired_speed) ** delta - (s_star / s) ** 2)
        else:
            return a * (1 - (self.velocity / self.desired_speed) ** delta)

    def decide_lane_change(self, surrounding_vehicle):
        """Decide whether to change lane based on MOBIL model."""

        if not self.lane_change:
            return False

        if surrounding_vehicle is None:
            return False
        
        incentive_to_change = self.idm_acceleration(None) - self.idm_acceleration(surrounding_vehicle)

        return incentive_to_change > lane_change_threshold

    def execute_lane_change(self):
        """Perform lane change over the specified duration."""
        
        if self.lane >= 1:
            self.dy = 0
        else:
            self.dy = 1 / self.lane_change_duration * TIME_STEP

    def update_lane_and_acceleration(self, surrounding_vehicle):
        """Update lane position and calculate acceleration based on surrounding vehicle."""
        if self.decide_lane_change(surrounding_vehicle):
            self.execute_lane_change()

        self.accel = self.idm_acceleration(
            surrounding_vehicle if surrounding_vehicle and self.lane == surrounding_vehicle.lane else None
        )

class Simulator:
    def __init__(self):
        self.simulation_data = []
        self.simulation_num = 0
        self.idx = 0

    def run(self):

        # Setting simulation data for different scenarios
        for _ in range(30):
            self.simulation_num += 1
            self._run_single_simulation(surrounding_vehicle_exists = True, ego_deisred_speed = 60/3.6, surrounding_vehicle_speed=30 / 3.6)

        for _ in range(30):
            self.simulation_num += 1
            self._run_single_simulation(surrounding_vehicle_exists = True, ego_deisred_speed = 60/3.6, surrounding_vehicle_speed=40 / 3.6)
        
        for _ in range(30):
            self.simulation_num += 1
            self._run_single_simulation(surrounding_vehicle_exists = False, ego_deisred_speed = 60/3.6)
        
        for _ in range(30):
            self.simulation_num += 1
            self._run_single_simulation(surrounding_vehicle_exists = False, ego_deisred_speed = 30/3.6)

        for _ in range(30):
            self.simulation_num += 1
            self._run_single_simulation(surrounding_vehicle_exists = True, ego_deisred_speed = 30/3.6, surrounding_vehicle_speed=40 / 3.6)

        # Convert the collected data into a DataFrame and save to a CSV file
        df = pd.DataFrame(self.simulation_data)
        df.to_csv("lane_change_simulations_data.csv", index=False)

    def _run_single_simulation(self, surrounding_vehicle_exists=True, ego_deisred_speed = 60/3.6, surrounding_vehicle_speed=30 / 3.6):

        # Initialize the vehicles for the simulation
        if surrounding_vehicle_exists:
            surrounding_vehicle = Vehicle(lane=0, position=50, velocity=surrounding_vehicle_speed, desired_speed=surrounding_vehicle_speed)
        else:
            surrounding_vehicle = None

        ego_vehicle = Vehicle(lane=0, position=0, velocity=0, desired_speed=ego_deisred_speed, lane_change = False)  # Using 60 km/h as desired speed

        time_points = np.arange(0, SIMULATION_TIME, TIME_STEP)

        for t in time_points:
            if surrounding_vehicle is not None:
                surrounding_vehicle.update()

            # Update lane position and calculate acceleration for the ego vehicle
            ego_vehicle.update_lane_and_acceleration(surrounding_vehicle)
            ego_vehicle.update()

            # Save simulation data for both vehicles at each time step
            self.idx += 1
            self.simulation_data.append({
                'idx': self.idx,
                'simulation_num': self.simulation_num,
                'time': t,
                'ego_x': ego_vehicle.last_position,
                'ego_y': ego_vehicle.last_lane,
                'ego_speed': ego_vehicle.last_velocity,
                'ego_desired_speed': ego_vehicle.desired_speed,
                'ego_accel': ego_vehicle.accel,
                'ego_dy': ego_vehicle.dy,
                'surrounding_agent_exist': 1 if surrounding_vehicle is not None else 0,
                'surrounding_x': surrounding_vehicle.last_position if surrounding_vehicle is not None else 0,
                'surrounding_y': surrounding_vehicle.last_lane if surrounding_vehicle is not None else 0,
                'surrounding_speed': surrounding_vehicle.last_velocity if surrounding_vehicle is not None else 0
            })

def visualize_multilane_driving_with_time():
    df = pd.read_csv("lane_change_simulations_data.csv")
    df_sim1 = df[df['simulation_num'] == 1]

    fig, axs = plt.subplots(4, 1, figsize=(10, 16))

    # Position along the road vs time
    axs[0].plot(df_sim1['time'], df_sim1['surrounding_x'], label='Surrounding Vehicle')
    axs[0].plot(df_sim1['time'], df_sim1['ego_x'], label='Ego Vehicle')
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("X (Position along road) [m]")
    axs[0].legend()
    axs[0].set_title("X Position vs Time")

    # Lane position vs time
    axs[1].plot(df_sim1['time'], df_sim1['surrounding_y'], label='Surrounding Vehicle')
    axs[1].plot(df_sim1['time'], df_sim1['ego_y'], label='Ego Vehicle')
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Y (Lane Position)")
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['Right Lane', 'Left Lane'])
    axs[1].legend()
    axs[1].set_title("Y Position (Lane) vs Time")

    # Speed vs time
    axs[2].plot(df_sim1['time'], df_sim1['surrounding_speed'] * 3.6, label='Surrounding Vehicle')
    axs[2].plot(df_sim1['time'], df_sim1['ego_speed'] * 3.6, label='Ego Vehicle')
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Speed [km/h]")
    axs[2].legend()
    axs[2].set_title("Speed vs Time")

    # Acceleration of the ego vehicle vs time
    axs[3].plot(df_sim1['time'], df_sim1['ego_accel'], label='Ego Vehicle Acceleration')
    axs[3].set_xlabel("Time [s]")
    axs[3].set_ylabel("Acceleration [m/s^2]")
    axs[3].legend()
    axs[3].set_title("Ego Vehicle Acceleration vs Time")

    plt.tight_layout()
    plt.savefig("multilane_driving_simulation_with_time.png")
    plt.show()

if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()
    visualize_multilane_driving_with_time()
