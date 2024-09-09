from NA_MEMD import *

# Future Updates
#   - Energy Criterion
#   - linear interpolation (interp1d) option
#   - speed-up using numba
#   - potentially create GUI for easy user usability


def main():

    # Generate a sample multivariate signal (3-channel signal)
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.array([
        np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.random.randn(1000),
        np.cos(2 * np.pi * 0.5 * t) + 0.3 * np.random.randn(1000),
        np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.random.randn(1000),
        np.cos(2 * np.pi * 0.5 * t) + 0.5 * np.random.randn(1000),
        np.cos(2 * np.pi * 0.5 * t) + 0.8 * np.random.randn(1000)

    ])

    # Visualize generated signal
    vis_signal(signal)

    # Example 1: Perform NA-MEMD on the signal
    imfs_na_memd = na_memd(signal)

    # Example 2: Perform standard MEMD on the signal
    imfs_memd = memd(signal)

    # Example 3: Visualize the results using the visualization function
    vis_imfs(imfs_na_memd)
    vis_imfs(imfs_memd)


if __name__ == "__main__":
    main()
