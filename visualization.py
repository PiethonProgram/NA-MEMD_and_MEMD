import matplotlib.pyplot as plt

def viz(imfs, num_samples=None):
    num_channels, num_imfs, num_data_points = imfs.shape
    print(num_channels, num_imfs, num_data_points)

    if num_samples is None or num_samples > num_data_points:
        num_samples = num_data_points

    fig, axes = plt.subplots(num_channels, num_imfs, figsize=(2 * num_imfs, 3 * num_channels), constrained_layout=True)
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
