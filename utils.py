import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.morphology

# Define the function to compute the boundaries based on a mathematical morphology
def computeBoundaries(signal, th):
    locs = np.where(signal > th)[0]
    indexes = np.zeros((signal.shape[0], 1))
    indexes[locs] = 1

    # Morphological operations
    selem = np.ones((70, 1)) #70
    closed = skimage.morphology.closing(indexes, selem)
    opened = skimage.morphology.opening(closed, selem)

    locsNew = np.where(opened == 1)[0]
    derNew = np.diff(locsNew)
    ind = np.where(derNew > 1)[0]

    ending = np.append(locsNew[ind], locsNew[-1])
    starting = np.append(locsNew[0], locsNew[ind + 1])

    return starting, ending

def plot_and_segment_cube_signals(base_path, params_dict, cube_number, segmented_data_dict, threshold, plot_signals=True):
    cube_name = f'Cube{cube_number}'
    cube_path = os.path.join(base_path, cube_name)

    sampling_rate = 200000  # 200,000 Hz
    time_step = 1 / sampling_rate * 1000  # Time step in ms

    if plot_signals:
        # 10 conditions, with 2 subplots (for channels 0 and 1) per condition
        fig, axes = plt.subplots(10, 2, figsize=(12, 40))
        fig.suptitle(f'{cube_name} Signals with Segment Markers', fontsize=16)

    for i in range(10):  # For each condition
        # Load channel data
        data_channel_0 = pd.read_csv(os.path.join(cube_path, 'channel_0', f'File_{i}.csv')).to_numpy().flatten()
        data_channel_1 = pd.read_csv(os.path.join(cube_path, 'channel_1', f'File_{i}.csv')).to_numpy().flatten()

        # Calculate time array for plotting
        time_array = np.arange(data_channel_0.size) * time_step

        # Use channel 1 to find boundaries
        starting_1, ending_1 = computeBoundaries(data_channel_0, threshold)
        num_segments = len(starting_1)
        segment_sizes = ending_1 - starting_1
        avg_size = np.mean(segment_sizes)
        std_dev = np.std(segment_sizes)

        # Print the statistics for the segments
        print(f'Condition {i+1}: Found {num_segments} segments, Avg. Size: {avg_size:.2f}, Std. Dev: {std_dev:.2f}')
        
        param_set = params_dict[cube_name][i]
        
        if plot_signals:
            # Plot signals with markers
            for j, (channel_data, time) in enumerate([(data_channel_0, time_array), (data_channel_1, time_array)]):
                ax = axes[i, j]
                ax.plot(time, channel_data)
                title = f'Cond {i+1}: {int(param_set["Power (W)"])}W, {int(param_set["Speed (mm/s)"])}mm/s'
                ax.set_title(title)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(f'Signal - channel_{j}')
                # Add markers for the start and end of each segment
                for start in starting_1:
                    ax.axvline(x=time_array[start], color='green', linestyle='--')
                for end in ending_1:
                    ax.axvline(x=time_array[end], color='red', linestyle='--')

        # Segment both channels using the computed boundaries and organize the data into a tensor
        min_size = min(segment_sizes)
        max_size = max(segment_sizes)
        print(f"Segment Max Size: {max_size}, Segment Min Size: {min_size}")
        data_tensor = np.zeros((num_segments, 2, min_size))
        for seg_index, start in enumerate(starting_1):
            end = start + min_size
            data_tensor[seg_index, 0, :] = data_channel_0[start:end]
            data_tensor[seg_index, 1, :] = data_channel_1[start:end]

        # Store in the dictionary with parameters as keys
        key = (param_set['Power (W)'], param_set['Speed (mm/s)'], cube_number)
        segmented_data_dict[key] = data_tensor

    if plot_signals:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

