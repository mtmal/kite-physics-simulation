from src.simulation import KiteSimulation

def main():
    # Create and run simulation with infinite tether
    sim = KiteSimulation(
        duration=10.0,
        time_step=0.1,
        visualization_update_freq=5,  # Update visualization every 5 physics steps
        wind_speed=27.78,  # 100 km/h
        kite_area=20.0,
        initial_height=100.0,  # Start at 100m height
        initial_velocity=10.0,  # Start with some forward velocity
        unroll_speed_factor=0.3,  # Use 30% of wind speed for unrolling
        power_efficiency=0.8,  # 80% power generation efficiency
        enable_turbulence=False,
        initial_tether_length=None,  # Infinite tether
        backend='numpy',
        restrict_lateral=False
    )
    
    sim.run()

if __name__ == "__main__":
    main() 