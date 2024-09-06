from NA_MEMD import *
from visualization import viz, vis_signal

# Future Updates
#   - Energy Criterion, Residual Criterion
#   - linear interpolation (interp1d) option
#   - speed-up using numba
#   - general speedup of programs
#   - Alternative EMD options
#   - improve visualizations
#   - potentially create GUI for easy user usability

def main():

    # Generate a sample multivariate signal (3-channel signal)
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.array([
        np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.random.randn(1000),
        np.cos(2 * np.pi * 0.5 * t) + 0.3 * np.random.randn(1000),
        np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.random.randn(1000)
    ])

    #Visualize generated signal
    vis_signal(signal)

    # Example 1: Perform NA-MEMD on the signal
    imfs_na_memd = na_memd(signal, n_dir=64, stop_crit='stop', intensity=0.1, na_method='na_fix')

    # Example 2: Perform standard MEMD on the signal
    imfs_memd = memd(signal, n_dir=64, stop_crit='stop')

    # Example 3: Visualize the results using the visualization function
    viz(imfs_na_memd, num_samples=signal.shape[1])
    viz(imfs_memd, num_samples=signal.shape[1])


    #vis_signal(add_noise(signal, method='na_var'))


if __name__ == "__main__":
    main()
