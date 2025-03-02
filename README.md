# Kite Physics Simulation

A 3D physics simulation of a tethered kite in wind conditions, with support for both CPU and GPU acceleration. Written with Claude 3.5 Sonnet.

## Features

- 3D visualization of kite movement
- Real-time force visualization
- Wind speed variation with height
- Optional wind turbulence
- Tether constraints using spring forces
- GPU acceleration support (using CuPy)
- Real-time force magnitude display
- Animation export capability

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
    height=300.0,          # initial height (meters)
    kite_area=20.0,        # kite area (m²)
    kite_mass=50.0,        # mass (kg)
    lift_coefficient=1.2,   # lift coefficient
    drag_coefficient=0.1,   # drag coefficient
    
    # Wind parameters
    wind_speed=27.78,      # reference wind speed (m/s)
    reference_height=10.0,  # height for reference wind speed (m)
    wind_shear_exponent=0.14, # power law exponent for wind profile
    enable_turbulence=True,# enable wind turbulence
    turbulence_intensity=0.1, # turbulence intensity
    
    # Tether parameters
    tether_length=400.0,   # tether length (m)
    tether_spring_constant=1000.0,  # tether stiffness (N/m)
    
    # Physics parameters
    gravity=9.81,          # gravitational acceleration (m/s²)
    max_velocity=100.0,    # maximum allowed velocity (m/s)
    max_force=1e4,         # force clipping threshold (N)
    
    # Simulation options
    backend='cupy',        # 'numpy' for CPU, 'cupy' for GPU
    restrict_lateral=False # allow lateral movement
)
```

## Physics Model

The simulation includes:
- Aerodynamic lift and drag forces
- Wind speed variation with height (power law profile)
- Optional wind turbulence
- Tether forces using spring model
- Gravity
- Kinematic updates using equations:
  - v = v₀ + at
  - p = p₀ + v₀t + ½at²

## Visualization

The simulation provides:
- Real-time 3D visualization
- Force vector display
- Force magnitude in Newtons
- Tether visualization
- MP4 animation export

## Dependencies

- numpy: Array operations (CPU)
- cupy: GPU acceleration
- matplotlib: 3D visualization
- tqdm: Progress bar
- ffmpeg: Animation export
