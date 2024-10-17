import numpy as np
import matplotlib.pyplot as plt

class Interp:
    """Interpolates"""

    def __init__(self, t_max, Nt):
        """Initialize constants"""
        self.t_max = t_max
        self.Nt = Nt

    def interpolate(self, t_non_uniform, u_non_uniform):
        """Piecewise linear interpolation"""
        t_uniform = np.linspace(0.0, self.t_max, self.Nt)
        u_interp = np.array([np.interp(t_uniform, t_non_uniform, u_t) for u_t in u_non_uniform.T]).T
        return u_interp
  
if __name__ == "__main__":
    from state_equation import StateEquation
    from Grid1D import Grid1D
    from macCormack_stepper import MacCormackStepper

    # Loop over grid size
    for nx in (8, 16, 32, 64, 128, 256, 512, 1024):
        grid = Grid1D(nx=nx)
        model = StateEquation()

        stepper = MacCormackStepper(grid=grid, model=model)

        # Set initial value to that of the known solution 
        t_init = 0.0
        uInit = model.exact_solution(grid.X, t_init)

        # Run the simulation from t_init up to t_final
        t_final = 5.0
        stepper.run(t_init, t_final, uInit)

        # Extract time steps and solutions
        t_non_uniform = np.array([time[0] for time in stepper.history])
        u_non_uniform = np.array([solution[1] for solution in stepper.history])

        # Flatten each timestep solution into 1D
        u_non_uniform = u_non_uniform.reshape(len(t_non_uniform), -1)  # Reshape into 2D array [timesteps, nx]

        t_max = np.pi / 2
        Nt = 1000  # Number of uniform time steps

        # Interpolation
        interp = Interp(t_max, Nt)
        u_even = interp.interpolate(t_non_uniform, u_non_uniform)

        # Ensure u_exact has the same shape as u_even
        t_uniform = np.linspace(0.0, t_max, Nt)
        u_exact = np.array([model.exact_solution(grid.X, t) for t in t_uniform])

        # Error calculation (make sure shapes match for comparison)
        err = u_exact - u_even
        err_norm = np.linalg.norm(err, 1) / nx

        print(f"Grid size: {nx}, Error norm: {err_norm}")

        # Plotting the results
        plt.figure(figsize=(15, 8))
        plt.plot(t_uniform, u_even[:, 0], label="Interpolated Solution")
        plt.plot(t_uniform, u_exact[:, 0], label="Exact Solution")
        
        # Adding labels and title
        plt.title(f'Exact vs Interpolated Solutions (Grid size: {nx})')
        plt.xlabel('Time (t)')
        plt.ylabel('Solution (u)')
        
        # Adding legend
        plt.legend()
        
        # Show plot
        plt.show()
