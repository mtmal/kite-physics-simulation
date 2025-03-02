from typing import Literal
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class KitePhysics:
    """
    Physics engine for kite simulation with support for both CPU (NumPy) and GPU (CuPy) computations.
    
    This class handles:
    - Wind speed calculations at different heights
    - Force calculations (lift, drag, spring forces)
    - Position and velocity updates
    - Optional turbulence simulation
    """
    
    def __init__(self, 
                 height: float = 300.0,          # meters
                 wind_speed: float = 27.78,      # m/s (100 km/h)
                 kite_area: float = 20.0,        # m²
                 air_density: float = 1.225,     # kg/m³
                 lift_coefficient: float = 1.2,   
                 drag_coefficient: float = 0.1,   
                 kite_mass: float = 50.0,        # kg
                 tether_length: float = 400.0,   # m
                 tether_spring_constant: float = 1000.0,  # N/m
                 reference_height: float = 10.0,  # m (height for reference wind speed)
                 wind_shear_exponent: float = 0.14,  # Power law exponent for wind profile
                 gravity: float = 9.81,          # m/s²
                 max_velocity: float = 100.0,    # m/s
                 max_force: float = 1e4,         # N
                 restrict_lateral: bool = True,
                 enable_turbulence: bool = False,
                 turbulence_intensity: float = 0.1,
                 backend: Literal['numpy', 'cupy'] = 'numpy'):
        """
        Initialize the physics engine.
        
        Args:
            height: Initial height of the kite
            wind_speed: Reference wind speed at reference height
            kite_area: Area of the kite for force calculations
            air_density: Air density for aerodynamic calculations
            lift_coefficient: Aerodynamic lift coefficient
            drag_coefficient: Aerodynamic drag coefficient
            kite_mass: Mass of the kite
            tether_length: Length of the tether
            tether_spring_constant: Spring constant for tether force calculation
            reference_height: Height at which wind_speed is measured
            wind_shear_exponent: Alpha value for power law wind profile
            gravity: Gravitational acceleration
            max_velocity: Maximum allowed velocity magnitude
            max_force: Force clipping threshold
            restrict_lateral: Whether to restrict movement in z-direction
            enable_turbulence: Whether to simulate wind turbulence
            turbulence_intensity: Intensity of wind turbulence
            backend: Computation backend ('numpy' for CPU, 'cupy' for GPU)
        """
        
        # Fall back to NumPy if CuPy is not available
        if backend == 'cupy' and not CUPY_AVAILABLE:
            print("Warning: CuPy not available, falling back to NumPy")
            backend = 'numpy'
        
        # Select computation backend
        self.xp = cp if backend == 'cupy' else np
        
        # Convert all parameters to float32 and move to appropriate device
        self.height = self.xp.float32(height)
        self.wind_speed = self.xp.float32(wind_speed)
        self.kite_area = self.xp.float32(kite_area)
        self.air_density = self.xp.float32(air_density)
        self.lift_coefficient = self.xp.float32(lift_coefficient)
        self.drag_coefficient = self.xp.float32(drag_coefficient)
        self.kite_mass = self.xp.float32(kite_mass)
        self.tether_length = self.xp.float32(tether_length)
        self.tether_spring_constant = self.xp.float32(tether_spring_constant)
        self.restrict_lateral = restrict_lateral
        self.enable_turbulence = enable_turbulence
        self.turbulence_intensity = self.xp.float32(turbulence_intensity)
        self.reference_height = self.xp.float32(reference_height)
        self.wind_shear_exponent = self.xp.float32(wind_shear_exponent)
        self.gravity = self.xp.float32(gravity)
        self.max_velocity = self.xp.float32(max_velocity)
        self.max_force = self.xp.float32(max_force)
        
        # Initialize state vectors
        self.position = self.xp.array([0.0, height, 0.0], dtype=self.xp.float32)
        self.velocity = self.xp.array([0.0, 0.0, 0.0], dtype=self.xp.float32)
        
    def calculate_wind_speed(self, height):
        """Calculate wind speed at a given height using power law profile."""
        wind_speed = self.wind_speed * (height/self.reference_height)**self.wind_shear_exponent
        
        if self.enable_turbulence:
            turbulence = self.xp.random.normal(0, self.turbulence_intensity * wind_speed)
            wind_speed += turbulence
            
        return wind_speed
    
    def calculate_forces(self):
        """
        Calculate all forces acting on the kite.
        
        Returns:
            tuple: (total_force, lift_force, drag_force)
                  Forces are converted to CPU if using GPU backend
        """
        # Calculate wind speed and relative wind vector
        current_wind = self.calculate_wind_speed(self.position[1])
        relative_wind = self.xp.array([float(current_wind), 0.0, 0.0], dtype=self.xp.float32) - self.velocity
        
        # Calculate wind direction, handling zero velocity case
        wind_speed_magnitude = self.xp.linalg.norm(relative_wind)
        if wind_speed_magnitude < 1e-6:
            wind_direction = self.xp.array([1.0, 0.0, 0.0], dtype=self.xp.float32)
        else:
            wind_direction = relative_wind / wind_speed_magnitude
        
        # Calculate aerodynamic forces
        dynamic_pressure = self.xp.maximum(0.0, 0.5 * self.air_density * wind_speed_magnitude**2)
        lift_force = dynamic_pressure * self.kite_area * self.lift_coefficient
        drag_force = dynamic_pressure * self.kite_area * self.drag_coefficient
        
        # Calculate tether spring force
        current_length = self.xp.linalg.norm(self.position)
        if current_length > self.tether_length:
            spring_force = -self.tether_spring_constant * (current_length - self.tether_length) * self.position / current_length
        else:
            spring_force = self.xp.zeros(3, dtype=self.xp.float32)
        
        # Combine all forces
        drag_x = self.xp.clip(drag_force * wind_direction[0], -self.max_force, self.max_force)
        lift_y = self.xp.clip(lift_force - self.kite_mass * self.gravity, -self.max_force, self.max_force)
        drag_z = self.xp.where(
            self.xp.array(self.restrict_lateral, dtype=bool),
            self.xp.zeros(1, dtype=self.xp.float32)[0],
            self.xp.clip(drag_force * wind_direction[2], -self.max_force, self.max_force)
        )
        
        total_force = self.xp.stack([drag_x, lift_y, drag_z]) + spring_force
        
        # Convert to CPU if using GPU
        if self.xp is cp:
            return (cp.asnumpy(total_force), float(lift_force), float(drag_force))
        return (total_force, float(lift_force), float(drag_force))
    
    def update_state(self, dt):
        """
        Update position and velocity based on forces using kinematic equations.
        
        Uses the equations:
        v = v0 + a*t
        p = p0 + v0*t + (1/2)a*t^2
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Updated position (converted to CPU if using GPU)
        """
        forces = self.calculate_forces()
        total_force = self.xp.asarray(forces[0], dtype=self.xp.float32)
        
        # Calculate acceleration
        acceleration = total_force / self.kite_mass
        
        # Store initial velocity for position update
        initial_velocity = self.velocity.copy()
        
        # Update velocity (v = v0 + a*t)
        self.velocity = self.xp.clip(
            initial_velocity + acceleration * dt,
            -self.max_velocity,
            self.max_velocity
        )
        
        # Update position using kinematic equation (p = p0 + v0*t + 0.5*a*t^2)
        self.position += initial_velocity * dt + 0.5 * acceleration * dt * dt
        
        # Convert position to CPU if using GPU
        return cp.asnumpy(self.position) if self.xp is cp else self.position 