import numpy as np
from .physics_base import KitePhysics, CUPY_AVAILABLE
from .visualization import KiteVisualizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import cupy as cp

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
                 visualization_update_freq=5,
                 **physics_params):
        """
        Initialize the simulation.
        
        Args:
            duration (float): Total simulation time in seconds
            time_step (float): Time step for physics calculations in seconds
            backend (str): Computation backend - 'numpy' for CPU or 'cupy' for GPU
            visualization_update_freq: Number of physics steps between visualization updates
            **physics_params: Additional parameters passed to KitePhysics
                            (e.g., height, wind_speed, kite_area, etc.)
        """
        self.duration = duration
        self.time_step = time_step
        self.visualization_update_freq = visualization_update_freq
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
        # Setup visualization with initial margins
        # Use a reasonable initial height for visualization (can be adjusted during simulation)
        initial_viz_height = 500.0  # meters
        self.visualizer.setup_plot(
            max_height=initial_viz_height,
            tether_length=float(self.physics.current_tether_length)
        )
        
        # Calculate total number of simulation steps
        total_steps = int(self.duration / self.time_step)
        
        # Create progress bar for monitoring simulation progress
        with tqdm(total=total_steps, desc="Simulating", unit="steps") as pbar:
            time = 0.0
            max_height_seen = 0.0
            step_count = 0
            
            while time < self.duration:
                # Update physics and get new state
                state = self.physics.update_state(self.time_step)
                
                # Update visualization less frequently
                if step_count % self.visualization_update_freq == 0:
                    # Convert GPU data to CPU only when needed for visualization
                    if CUPY_AVAILABLE and isinstance(state['position'], cp.ndarray):
                        position = cp.asnumpy(state['position'])
                        force = cp.asnumpy(state['force'])
                        power = float(state['power'])
                    else:
                        position = state['position']
                        force = state['force']
                        power = float(state['power'])
                    
                    # Track maximum height for visualization scaling
                    current_height = float(position[1])
                    if current_height > max_height_seen:
                        max_height_seen = current_height
                        # Update plot limits if height increases significantly
                        self.visualizer.update_height_limit(max_height_seen * 1.2)
                    
                    # Update visualization
                    self.visualizer.update_frame(
                        position=position,
                        forces=(force, 0, 0),  # Only passing total force for now
                        power=power
                    )
                    
                    # Update matplotlib display less frequently
                    plt.pause(0.001)
                
                # Advance simulation time
                time += self.time_step
                step_count += 1
                pbar.update(1)
        
        # Save the animation after simulation is complete
        print("\nSaving animation...")
        self.visualizer.save_animation()
        print("Simulation complete!") 