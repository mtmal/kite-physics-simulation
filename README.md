# Kite Physics Simulation

A 3D physics simulation of a power-generating airborne wind energy system, with support for both CPU and GPU acceleration.

## Features

### Physics Model
- **Wind Profile**: Power law wind shear model with configurable reference height and exponent
- **Aerodynamics**: 
  - Lift and drag forces with configurable coefficients
  - Wind speed variation with height
  - Optional wind turbulence
- **Tether System**: 
  - Option for infinite tether length (continuous unrolling)
  - Power generation through tether unrolling
- **Power Generation**:
  - Based on tether tension and unroll speed
  - Power = Tension × Unroll Speed × Efficiency
  - Unroll speed = k × W × cos(θ)
    - k: unroll speed factor (0-1)
    - W: wind speed at kite height
    - θ: angle between tether and horizontal
  - Configurable power generation efficiency

### Technical Features
- Real-time 3D visualization
- Force vector display
- GPU acceleration support (using CuPy)
- Real-time power generation display
- Animation export capability
- Configurable visualization update frequency

## Requirements

- Python 3.8 or higher
- CUDA toolkit (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kite-simulation.git
cd kite-simulation
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```

3. Install the package:
   
   - For CPU-only version:
   ```bash
   pip install -e .
   ```
   
   - For GPU-accelerated version:
   ```bash
   pip install -e ".[cuda]"
   ```

## Usage

Run the simulation:
```bash
python main.py
```

### Configuration

You can modify simulation parameters in `main.py`:

```python
sim = KiteSimulation(
    # Time parameters
    duration=10.0,          # simulation duration (seconds)
    time_step=0.1,         # physics time step (seconds)
    
    # Kite parameters
    initial_height=100.0,   # initial height (meters)
    initial_velocity=10.0,  # initial velocity (m/s)
    kite_area=20.0,        # kite area (m²)
    kite_mass=50.0,        # mass (kg)
    lift_coefficient=1.2,   # lift coefficient
    drag_coefficient=0.1,   # drag coefficient
    
    # Wind parameters
    wind_speed=27.78,      # reference wind speed (m/s)
    reference_height=10.0,  # height for reference wind speed (m)
    wind_shear_exponent=0.14, # power law exponent for wind profile
    enable_turbulence=False,# enable wind turbulence
    turbulence_intensity=0.1, # turbulence intensity
    
    # Power generation parameters
    initial_tether_length=None,  # None for infinite tether
    unroll_speed_factor=0.3,    # fraction of wind speed for unrolling
    power_efficiency=0.8,       # power generation efficiency
    
    # Physics parameters
    gravity=9.81,          # gravitational acceleration (m/s²)
    max_velocity=100.0,    # maximum allowed velocity (m/s)
    
    # Simulation options
    backend='numpy',       # 'numpy' for CPU, 'cupy' for GPU
    visualization_update_freq=5,  # update visualization every N steps
    restrict_lateral=False # allow lateral movement
)
```

## Physics Model

The simulation models:

### Wind
- Power law wind profile:
  - W(h) = W_ref × (h/h_ref)^α
  - W(h): Wind speed at height h
  - W_ref: Reference wind speed
  - h_ref: Reference height
  - α: Wind shear exponent

### Aerodynamics
- Dynamic pressure:
  - q = ½ρv²
  - ρ: Air density
  - v: Relative wind speed
- Lift force:
  - L = qAC_L
  - A: Kite area
  - C_L: Lift coefficient
- Drag force:
  - D = qAC_D
  - C_D: Drag coefficient

### Power Generation
- Tension:
  - T = F·r̂
  - F: Total aerodynamic force
  - r̂: Unit vector along tether
- Unroll speed:
  - v = kW cos(θ)
  - k: Unroll speed factor
  - W: Wind speed at kite height
  - θ: Angle between tether and horizontal
- Power:
  - P = Tvη
  - η: Power generation efficiency

### Kinematics
- Velocity update:
  - v = v₀ + at
- Position update:
  - p = p₀ + v₀t + ½at²
- Tether length update (for infinite tether):
  - L = L₀ + vt

## Visualization

The simulation provides:
- Real-time 3D visualization
- Force vector display
- Force magnitude in Newtons
- Power generation in kilowatts
- Tether visualization
- MP4 animation export

## Dependencies

Core dependencies:
- numpy: Array operations (CPU)
- matplotlib: 3D visualization
- tqdm: Progress bar

Optional dependencies:
- cupy: GPU acceleration
- ffmpeg: Animation export
