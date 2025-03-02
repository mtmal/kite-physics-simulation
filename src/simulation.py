import numpy as np
from .physics_base import KitePhysics
from .visualization import KiteVisualizer
from tqdm import tqdm
import matplotlib.pyplot as plt

class KiteSimulation:
    """
    Main simulation controller class that coordinates physics calculations and visualization.
    
    This class handles:
    - Initialization of physics engine and visualizer
    - Time stepping through the simulation
    - Progress tracking
    - Coordination between physics updates and visualization
    """
    
    def __init__(self, 
                 duration=10.0,     # seconds
                 time_step=0.1,     # seconds
                 backend='numpy',   # 'numpy' or 'cupy'
                 **physics_params):
        """
        Initialize the simulation.
        
        Args:
            duration (float): Total simulation time in seconds
            time_step (float): Time step for physics calculations in seconds
            backend (str): Computation backend - 'numpy' for CPU or 'cupy' for GPU
            **physics_params: Additional parameters passed to KitePhysics
                            (e.g., height, wind_speed, kite_area, etc.)
        """
        self.duration = duration
        self.time_step = time_step
        # Initialize physics engine with specified backend and parameters
        self.physics = KitePhysics(backend=backend, **physics_params)
        # Initialize visualization system
        self.visualizer = KiteVisualizer()
        
    def run(self):
        """
        Run the complete simulation.
        
        This method:
        1. Sets up the visualization
        2. Steps through time, updating physics and visualization
        3. Shows progress with a progress bar
        4. Saves the final animation
        """
        # Setup visualization with margins around max height
        self.visualizer.setup_plot(
            max_height=float(self.physics.height * 1.2),  # Add 20% margin
            tether_length=float(self.physics.tether_length)
        )
        
        # Calculate total number of simulation steps
        total_steps = int(self.duration / self.time_step)
        
        # Create progress bar for monitoring simulation progress
        with tqdm(total=total_steps, desc="Simulating", unit="steps") as pbar:
            time = 0.0
            while time < self.duration:
                # Update physics and get new position
                position = self.physics.update_state(self.time_step)
                # Calculate forces at current position
                forces = self.physics.calculate_forces()
                
                # Update visualization with new position and forces
                self.visualizer.update_frame(position, forces)
                
                # Advance simulation time
                time += self.time_step
                pbar.update(1)
                
                # Update matplotlib display and prevent memory issues
                plt.pause(0.001)
        
        # Save the animation after simulation is complete
        print("\nSaving animation...")
        self.visualizer.save_animation()
        print("Simulation complete!") 