from NA_MEMD import *

# Future Updates
#   - Energy Threshold Criterion (currently not working)
#   - linear interpolation (interp1d) option (currently not implemented)
#   - speed-up using numba (currently not implemented)
#   - potentially create GUI for easy user usability (planning phase)




def main():

    # Generate a sample multivariate signal (5-channel signal)
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

    # -----------------------------------------------------------------------------------

    # Adds noise to signal (default returns array of (Channel*2, Data Points))
    # Default behavior adds noise channels, effectively doubling the input channels
    noise_add = add_noise(signal)

    # -----------------------------------------------------------------------------------

    # NA-MEMD: Perform noise-assisted MEMD (returns matrix of (Channels, IMFs, Data Points))
    imfs_na_memd = na_memd(signal)

    # -----------------------------------------------------------------------------------

    # Standard MEMD: Perform regular MEMD (returns matrix of (Channels, IMFs, Data Points))
    imfs_memd = memd(signal)

    # -----------------------------------------------------------------------------------

    # Visualize NA-MEMD and MEMD results (channels on Y-axis, IMFs on X-axis)
    vis_imfs(imfs_na_memd)
    vis_imfs(imfs_memd)

    # -----------------------------------------------------------------------------------
    # NA-MEMD and MEMD Parameters Explained :
    # -----------------------------------------------------------------------------------

    """  MEMD Inputs :
    
    signal : input signal
    n_dir : number of projections
        default is 50
    stop_crit : stopping criterion
        'stop' : standard stopping criterion
        'fix_h' : number of consecutive iterations where extrema and zero crossing differences <= 1
            default is 'stop'
    stop-vec : stopping criteria (sd, sd2, tol) <= (standard dev, standard dev of envelope, tolerance)
        only valid when stop_crit == 'stop'
        default is (0.05, 0.5, 0.05)
    e_thresh : stopping criterion using energy threshold
        only valid when stop_crit == 'e_diff' 
        default is 1e-3
    n_iter : number of iterations
        only valid when stop_crit == 'fix_h'
        default is 3
    max_imf : maximum number of IMFs before stop
        default is 100
    """

    # -----------------------------------------------------------------------------------

    """ NA-MEMD Inputs : 
    Same options as MEMD + :
    
    na_method : noise addition option (see noise.py for full list)
        default is 'na_fix'
    intensity : intensity of noise
        default is 0.1
    add_rchannel : number of noise channels to add
        default is set to number of input channels
    output_condition : output IMFs of noise channels
        default is set to False
    """


if __name__ == "__main__":
    main()
