import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Switch to TkAgg backend for interactive plotting
plt.switch_backend('TkAgg')

class KiteVisualizer:
    """
    Visualization system for kite simulation.
    
    This class handles:
    - 3D visualization of kite position
    - Force vector visualization
    - Real-time force magnitude display
    - Animation recording and saving
    """
    
    def __init__(self):
        """Initialize the visualization system with matplotlib figure and 3D axes."""
        plt.ion()  # Enable interactive mode for real-time updates
        
        # Create figure and 3D axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize visualization elements
        self.lines = []
        self.force_arrows = []
        self.frames = []  # Store frames for animation
        
        # Initialize plot elements that will be updated
        self.kite_point = None
        self.tether_line = None
        self.force_arrow = None
        self.force_text = None
        self.power_text = None
        
    def setup_plot(self, max_height, tether_length):
        """
        Set up the 3D plot with appropriate axes limits and labels.
        
        Args:
            max_height (float): Maximum height for y-axis
            tether_length (float): Tether length for x and z axes scaling
        """
        # Set axis limits
        self.ax.set_xlim([-tether_length/2, tether_length/2])
        self.ax.set_ylim([0, max_height * 1.2])
        self.ax.set_zlim([-tether_length/2, tether_length/2])
        
        # Set axis labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Height (m)')
        self.ax.set_zlabel('Z (m)')
        
        # Initialize plot elements
        self.kite_point, = self.ax.plot([], [], [], 'ro', markersize=10)  # Red dot for kite
        self.tether_line, = self.ax.plot([], [], [], 'k-', linewidth=1)   # Black line for tether
        self.force_arrow = self.ax.quiver(0, 0, 0, 0, 0, 0)               # Force vector
        self.force_text = self.ax.text2D(0.02, 0.95, '',                  # Force magnitude text
                                        transform=self.ax.transAxes)
        
        # Add power text display
        self.power_text = self.ax.text2D(0.02, 0.90, '',  # Position below force text
                                        transform=self.ax.transAxes)
        
    def update_frame(self, position, forces, power=0.0):
        """
        Update the visualization for a new frame.
        
        Args:
            position (np.ndarray): Current position of the kite [x, y, z]
            forces (tuple): (total_force, lift_force, drag_force)
            power (float): Current power generation in watts
        """
        # Update kite position marker
        self.kite_point.set_data_3d([position[0]], [position[1]], [position[2]])
        
        # Update tether line (from origin to kite position)
        self.tether_line.set_data_3d([0, position[0]], [0, position[1]], [0, position[2]])
        
        # Update force vector visualization
        total_force = forces[0]
        scale = 0.01  # Reduced from 0.1 to 0.01 to make force vectors smaller
        if self.force_arrow:
            self.force_arrow.remove()
        self.force_arrow = self.ax.quiver(
            position[0], position[1], position[2],  # Start at kite position
            total_force[0] * scale, total_force[1] * scale, total_force[2] * scale,  # Scaled force components
            color='blue'
        )
        
        # Update force magnitude text
        force_magnitude = np.linalg.norm(total_force)
        self.force_text.set_text(f'Total Force: {force_magnitude:.1f} N')
        
        # Update power text
        self.power_text.set_text(f'Power Generated: {power/1000:.1f} kW')
        
        # Update display and store frame
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.frames.append(self.fig.canvas.copy_from_bbox(self.ax.bbox))
        
    def save_animation(self, filename='kite_simulation.mp4', fps=30):
        """
        Save the recorded frames as an animation.
        
        Args:
            filename (str): Output filename (default: 'kite_simulation.mp4')
            fps (int): Frames per second for the animation (default: 30)
        """
        anim = FuncAnimation(
            self.fig, 
            lambda frame: self.frames[frame], 
            frames=len(self.frames),
            interval=1000/fps
        )
        anim.save(filename, writer='ffmpeg')

    def update_height_limit(self, new_height):
        """Update the height limit of the plot."""
        self.ax.set_ylim([0, new_height])
        self.fig.canvas.draw() 