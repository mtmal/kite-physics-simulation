from typing import Literal, Optional
import numpy as np

# Define cp as None if import fails
cp = None
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
                 wind_speed: float = 27.78,      # m/s (100 km/h)
                 kite_area: float = 20.0,        # m²
                 air_density: float = 1.225,     # kg/m³
                 lift_coefficient: float = 1.2,   
                 drag_coefficient: float = 0.1,   
                 kite_mass: float = 50.0,        # kg
                 initial_tether_length: Optional[float] = 400.0,   # m, None for infinite
                 initial_height: float = 100.0,   # m, initial height to start simulation
                 initial_velocity: float = 10.0,  # m/s, initial velocity magnitude
                 unroll_speed_factor: float = 0.3,  # Fraction of wind speed for unrolling
                 power_efficiency: float = 0.8,   # Power generation efficiency
                 reference_height: float = 10.0,  # m (height for reference wind speed)
                 wind_shear_exponent: float = 0.14,  # Power law exponent for wind profile
                 gravity: float = 9.81,          # m/s²
                 max_velocity: float = 100.0,    # m/s
                 restrict_lateral: bool = True,
                 enable_turbulence: bool = False,
                 turbulence_intensity: float = 0.1,
                 backend: Literal['numpy', 'cupy'] = 'numpy'):
        """
        Initialize the physics engine.
        
        Args:
            wind_speed: Reference wind speed at reference height
            kite_area: Area of the kite for force calculations
            air_density: Air density for aerodynamic calculations
            lift_coefficient: Aerodynamic lift coefficient
            drag_coefficient: Aerodynamic drag coefficient
            kite_mass: Mass of the kite
            initial_tether_length: Initial length of the tether. If None, tether is infinite
            initial_height: Initial height to start simulation
            initial_velocity: Initial velocity magnitude
            unroll_speed_factor: Fraction of wind speed used for tether unrolling (0-1)
            power_efficiency: Efficiency of power generation system (0-1)
            reference_height: Height at which wind_speed is measured
            wind_shear_exponent: Alpha value for power law wind profile
            gravity: Gravitational acceleration
            max_velocity: Maximum allowed velocity magnitude
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
        self.xp = np if backend == 'numpy' or not CUPY_AVAILABLE else cp
        
        # Convert all parameters to float32 and move to appropriate device
        self.wind_speed = self.xp.float32(wind_speed)
        self.kite_area = self.xp.float32(kite_area)
        self.air_density = self.xp.float32(air_density)
        self.lift_coefficient = self.xp.float32(lift_coefficient)
        self.drag_coefficient = self.xp.float32(drag_coefficient)
        self.kite_mass = self.xp.float32(kite_mass)
        self.infinite_tether = initial_tether_length is None
        self.current_tether_length = self.xp.float32(initial_tether_length if initial_tether_length is not None else 1e6)
        self.tether_unroll_speed = self.xp.float32(0.0)  # m/s
        self.generated_power = self.xp.float32(0.0)  # Watts
        self.restrict_lateral = restrict_lateral
        self.enable_turbulence = enable_turbulence
        self.turbulence_intensity = self.xp.float32(turbulence_intensity)
        self.reference_height = self.xp.float32(reference_height)
        self.wind_shear_exponent = self.xp.float32(wind_shear_exponent)
        self.gravity = self.xp.float32(gravity)
        self.max_velocity = self.xp.float32(max_velocity)
        self.unroll_speed_factor = self.xp.float32(unroll_speed_factor)
        self.power_efficiency = self.xp.float32(power_efficiency)
        
        # Initialize state vectors with some initial height and velocity
        self.position = self.xp.array([0.0, initial_height, 0.0], dtype=self.xp.float32)
        # Give initial velocity in the x-direction to start the motion
        self.velocity = self.xp.array([initial_velocity, 0.0, 0.0], dtype=self.xp.float32)
        
    def calculate_wind_speed(self, height):
        """Calculate wind speed at a given height using power law profile."""
        wind_speed = self.wind_speed * (height/self.reference_height)**self.wind_shear_exponent
        
        if self.enable_turbulence:
            turbulence = self.xp.random.normal(0, self.turbulence_intensity * wind_speed)
            wind_speed += turbulence
            
        return wind_speed
    
    def calculate_forces(self):
        """Calculate all forces acting on the kite."""
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
        
        # Combine all forces (using fixed max_force value for numerical stability)
        MAX_FORCE = 1e4  # Fixed value
        drag_x = self.xp.clip(drag_force * wind_direction[0], -MAX_FORCE, MAX_FORCE)
        lift_y = self.xp.clip(lift_force - self.kite_mass * self.gravity, -MAX_FORCE, MAX_FORCE)
        
        # Convert boolean to array for CuPy compatibility
        restrict_lateral_array = self.xp.array(self.restrict_lateral, dtype=bool)
        drag_z = self.xp.where(
            restrict_lateral_array,
            self.xp.float32(0.0),
            self.xp.clip(drag_force * wind_direction[2], -MAX_FORCE, MAX_FORCE)
        )
        
        # Calculate total force (no need to index drag_z)
        total_force = self.xp.array([drag_x, lift_y, drag_z], dtype=self.xp.float32)
        
        # Calculate tension (radial component of forces)
        position_magnitude = self.xp.linalg.norm(self.position)
        if position_magnitude > 0:
            # Unit vector pointing from anchor to kite
            position_unit = self.position / position_magnitude
            
            # Calculate tension as projection of total aerodynamic forces onto tether direction
            # Positive tension means the kite is pulling outward
            tension = self.xp.dot(total_force, position_unit)
            
            # For infinite tether mode with positive tension:
            if self.infinite_tether and tension > 0:
                # Get wind speed at kite's current height using power law profile
                wind_speed = self.calculate_wind_speed(self.position[1])
                
                # Calculate cosine of angle between tether and horizontal (x-axis)
                # This represents how much of the wind's force contributes to unrolling
                # cos_theta = x_component of unit vector (adjacent/hypotenuse)
                cos_theta = self.xp.abs(position_unit[0])
                
                # Calculate unroll speed as fraction of projected wind speed
                # - wind_speed * cos_theta: wind speed component along tether direction
                # - unroll_speed_factor: controllable fraction (0-1) of that speed used for unrolling
                self.tether_unroll_speed = wind_speed * cos_theta * self.unroll_speed_factor
                
                # Calculate power generation:
                # P = F * v * η
                # where:
                # - F (tension) is the force along the tether
                # - v (tether_unroll_speed) is the speed of unrolling
                # - η (power_efficiency) accounts for conversion losses
                self.generated_power = tension * self.tether_unroll_speed * self.power_efficiency
            else:
                # No power generation when tension is zero or negative
                self.tether_unroll_speed = 0.0
                self.generated_power = 0.0
            
            # For finite tether, prevent movement beyond length
            if not self.infinite_tether and position_magnitude > self.current_tether_length:
                total_force = -total_force  # Prevent further movement
        else:
            total_force = self.xp.zeros(3, dtype=self.xp.float32)
            self.tether_unroll_speed = 0.0
            self.generated_power = 0.0
        
        # Don't convert to CPU here - keep data on GPU
        return (total_force, float(lift_force), float(drag_force))
    
    def update_state(self, dt):
        """Update position and velocity."""
        forces = self.calculate_forces()
        total_force = self.xp.asarray(forces[0], dtype=self.xp.float32)
        
        # Update tether length based on unroll speed
        if self.infinite_tether:
            self.current_tether_length += self.tether_unroll_speed * dt
        
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
        
        # Return state without converting to CPU
        return {
            'position': self.position,
            'velocity': self.velocity,
            'force': total_force,
            'power': self.generated_power,
            'tether_length': self.current_tether_length
        } 