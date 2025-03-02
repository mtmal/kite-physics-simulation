from src.simulation import KiteSimulation

def main():
    # Create and run simulation with default parameters
    sim = KiteSimulation(
        duration=10.0,  # reduced from 100s to 10s
        time_step=0.1,
        height=300.0,
        wind_speed=27.78,  # 100 km/h
        kite_area=20.0,
        enable_turbulence=True,
        tether_spring_constant=1000.0,  # N/m
        backend='cupy',  # or 'numpy' for CPU-only
        restrict_lateral=False
    )
    
    sim.run()

if __name__ == "__main__":
    main() 