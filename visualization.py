import matplotlib.pyplot as plt
import numpy as np

def viz(imfs, num_samples=500):
    num_channels, num_imfs, num_data_points = imfs.shape
    print(num_channels, num_imfs, num_data_points)

    if num_samples is None or num_samples > num_data_points:
        num_samples = num_data_points

    fig, axes = plt.subplots(num_channels, num_imfs, figsize=(5 * num_imfs, 3 * num_channels), constrained_layout=True)
    fig.suptitle("Channel IMFs", fontsize=20)

    for channel_index in range(num_channels):
        for i in range(num_imfs):
            ax = axes[channel_index, i] if num_imfs > 1 else axes[channel_index]
            ax.plot(imfs[channel_index, i, :num_samples])
            if i == 0:
                ax.set_ylabel(f'Channel {channel_index + 1}')
            if channel_index == 0:
                ax.set_title(f'IMF {i + 1}' if i < (num_imfs - 1) else 'Residual')
            ax.set_xlim(0, num_samples)
            ax.grid(True)

    plt.show()

def vis_signal(signal):
    fig, ax = plt.subplots(figsize=(10, 6))
    num_channels, num_data_points = signal.shape
    t = np.arange(num_data_points)

    for channel_index in range(num_channels):
        ax.plot(t, signal[channel_index, :], label=f'Channel {channel_index + 1}')

    ax.set_title("Original Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()

    plt.show()
