import os
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.signal import detrend
from scipy.signal import hilbert
from scipy.signal import cwt, morlet2


def get_imu_names(h5_path):
    """Returns the IMU names under the 'IMU' group in an HDF5 file, if the file exists."""
    if not os.path.isfile(h5_path):
        print(f"File not found: {h5_path}")
        return []

    with h5py.File(h5_path, "r") as file:
        if "IMU" in file:
            return list(file["IMU"].keys())
        else:
            print("'IMU' group not found in the file.")
            return []


def get_imu_keys(h5_path, imu_name):
    """Returns the keys inside a specific IMU group, if both the file and IMU exist."""
    if not os.path.isfile(h5_path):
        print(f"File not found: {h5_path}")
        return []

    with h5py.File(h5_path, "r") as file:
        if "IMU" in file and imu_name in file["IMU"]:
            return list(file["IMU"][imu_name].keys())
        else:
            print(f"IMU '{imu_name}' not found in the file.")
            return []


def load_imu_data(path, imu_key="DOT_40195BFD802900B5"):
    """Load IMU data from HDF5 file and convert timestamps to relative seconds."""
    with h5py.File(path, "r") as file:
        imu = file["IMU"][imu_key]
        acc = imu["Accelerometer"][:]  # shape (3, N)
        gyro = imu["Gyroscope"][:]  # shape (3, N)
        orient = imu["Orientation"][:]  # shape (3, N)
        time = imu["Timestamp_Sensor"][:]  # shape (N,)

    return acc, gyro, orient, time


def standardize_time(time_raw):
    """Convert raw timestamps (e.g. UNIX ns, ms) to seconds starting at 0."""
    time = np.array(time_raw, dtype=np.float64)

    if time.max() > 1e12:
        time = time / 1e9
    elif time.max() > 1e9:
        time = time / 1e6
    elif time.max() > 1e6:
        time = time / 1e3

    return time - time[0]


def plot_imu_data(acc, gyro, orient, time):
    """Plot IMU data: Accelerometer, Gyroscope, and Orientation."""
    time = standardize_time(time)
    # Accelerometer
    plt.figure(figsize=(10, 4))
    plt.plot(time, acc[0], label="Acc X")
    plt.plot(time, acc[1], label="Acc Y")
    plt.plot(time, acc[2], label="Acc Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Accelerometer Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gyroscope
    plt.figure(figsize=(10, 4))
    plt.plot(time, gyro[0], label="Gyro X")
    plt.plot(time, gyro[1], label="Gyro Y")
    plt.plot(time, gyro[2], label="Gyro Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (°/s)")
    plt.title("Gyroscope Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Orientation
    plt.figure(figsize=(10, 4))
    plt.plot(time, orient[0], label="Orientation X")
    plt.plot(time, orient[1], label="Orientation Y")
    plt.plot(time, orient[2], label="Orientation Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees")
    plt.title("Orientation Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def preprocess_signal(signal):
    """
    Removes linear trend and centers the signal.
    """
    signal = detrend(signal)  # Remove linear trend
    signal = signal - signal.mean()  # Center signal around 0
    return signal


def compute_fft(signal, fs):
    """
    Computes the FFT and returns frequency and amplitude.

    Parameters:
    - signal: 1D numpy array
    - fs: sampling frequency (Hz)

    Returns:
    - freqs: frequency axis
    - fft_vals: magnitude of FFT
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    return freqs, fft_vals


def plot_fft(freqs, fft_vals, title="FFT of Signal"):
    """
    Plots the frequency spectrum.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_vals, label="FFT magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def analyze_fft_with_harmonics(
    signal, fs, n_harmonics=3, title="FFT with Fundamental and Harmonics"
):
    """
    Preprocesses a signal, computes FFT, detects fundamental frequency,
    and plots the FFT with fundamental and harmonic markers.

    Parameters:
    - signal: 1D array-like, raw signal
    - fs: int, sampling frequency (Hz)
    - n_harmonics: int, number of harmonics to mark
    - title: str, title for the plot

    Returns:
    - fundamental_freq: float, detected fundamental frequency in Hz
    """
    # Preprocess signal
    signal = detrend(signal)
    signal = signal - signal.mean()

    # FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_vals = np.abs(np.fft.rfft(signal))

    # Ignore DC component
    fft_vals_no_dc = fft_vals.copy()
    fft_vals_no_dc[0] = 0

    # Find index of max peak
    peak_idx = np.argmax(fft_vals_no_dc)
    fundamental_freq = freqs[peak_idx]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_vals, label="FFT magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)

    # Mark fundamental and harmonics
    for i in range(1, n_harmonics + 1):
        harmonic_freq = i * fundamental_freq
        if harmonic_freq < freqs[-1]:
            plt.axvline(harmonic_freq, color="r", linestyle="--", alpha=0.7)
            plt.text(
                harmonic_freq,
                max(fft_vals) * 0.8,
                f"{i}×{fundamental_freq:.2f} Hz",
                rotation=90,
                color="r",
                ha="right",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.legend()
    plt.show()

    return fundamental_freq


def estimate_tremor_direction_pca(tremor_band, fs=60, window_sec=1.0):
    """
    Estimate tremor direction using PCA on bandpass-filtered acceleration data.

    Parameters:
        tremor_band (ndarray): shape (N, 3), filtered acceleration signal in world frame
        fs (int): sampling rate (Hz)
        window_sec (float): window size in seconds for PCA

    Returns:
        directions (ndarray): shape (N - window_size + 1, 3), dominant direction vectors
    """
    window = int(fs * window_sec)
    N = tremor_band.shape[0]
    directions = []

    for i in range(N - window):
        window_data = tremor_band[i : i + window]
        pca = PCA(n_components=1)
        pca.fit(window_data)
        dir_vec = pca.components_[0]
        directions.append(dir_vec)

    directions = np.array(directions)
    return directions


def plot_fft_cwt_xyz(
    signal_xyz, time, fs=60, max_freq=15, wavelet_width=6, signal_type="Acceleration"
):
    """
    Plot FFT and CWT spectrograms for X, Y, Z axes of a 3-axis signal with standardized colorbars and y-axis scaling.

    Parameters:
        signal_xyz (ndarray): shape (N, 3) = X, Y, Z axes.
        time (ndarray): time vector of length N
        fs (int): sampling rate in Hz
        max_freq (float): max frequency to display in CWT
        wavelet_width (float): wavelet resolution for CWT
        signal_type (str): title prefix (e.g., "Acceleration", "Velocity")
    """

    axes = ["X", "Y", "Z"]
    fft_vals_all = []
    cwt_amplitudes = []

    # Precompute FFT and CWT for all axes
    for i in range(3):
        signal = signal_xyz[:, i]
        freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
        fft_vals = np.abs(np.fft.rfft(signal))
        fft_vals_all.append((freqs, fft_vals))

        scales = np.arange(1, int(fs / 1.0))
        cwt_matrix = cwt(signal, morlet2, scales, w=wavelet_width)
        cwt_amplitudes.append(np.abs(cwt_matrix))

    # Standardize y-axis for FFT
    max_fft = max(np.max(fft[1]) for fft in fft_vals_all)
    max_cwt_amp = max(np.max(cwt_amp) for cwt_amp in cwt_amplitudes)

    # Plot per axis
    for i in range(3):
        freqs, fft_vals = fft_vals_all[i]
        cwt_amp = cwt_amplitudes[i]
        scales = np.arange(1, int(fs / 1.0))
        cwt_freqs = fs / (wavelet_width * scales)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # FFT plot
        axs[0].plot(freqs, fft_vals)
        axs[0].set_title(f"{signal_type} - FFT ({axes[i]}-axis)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_ylim(0, max_fft * 1.05)
        axs[0].grid(True)

        # CWT plot
        im = axs[1].imshow(
            cwt_amp,
            extent=[time[0], time[-1], cwt_freqs[-1], cwt_freqs[0]],
            aspect="auto",
            cmap="jet",
            origin="lower",
            vmin=0,
            vmax=max_cwt_amp,
        )
        axs[1].set_title(f"{signal_type} - CWT Spectrogram ({axes[i]}-axis)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Frequency (Hz)")
        axs[1].set_ylim(0, max_freq)
        plt.colorbar(im, ax=axs[1], label="Amplitude")

        plt.tight_layout()
        plt.show()


def analyze_tremor_hilbert(signal, time, axis_labels=None):
    """
    Applies the Hilbert transform to multi-axis tremor data and plots:
    - Tremor envelope (instantaneous amplitude)
    - Instantaneous phase
    - Phase portrait
    - Phase difference between axes (if 2 or more axes)

    Parameters:
    - signal: np.ndarray of shape (N, D), where N is time and D is number of axes (e.g., 3)
    - time: np.ndarray of shape (N,)
    - axis_labels: list of axis names, e.g. ['X', 'Y', 'Z']
    """

    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if axis_labels is None:
        axis_labels = [f"Axis {i}" for i in range(signal.shape[1])]

    analytic_signal = hilbert(signal, axis=0)
    envelope = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal), axis=0)

    # Plot envelope
    plt.figure(figsize=(10, 4))
    for i in range(signal.shape[1]):
        plt.plot(time, envelope[:, i], label=f"{axis_labels[i]} Envelope")
    plt.title("Tremor Envelope (Hilbert Amplitude)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot phase
    plt.figure(figsize=(10, 4))
    for i in range(signal.shape[1]):
        plt.plot(time, phase[:, i], label=f"{axis_labels[i]} Phase")
    plt.title("Instantaneous Phase")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Phase portrait
    if signal.shape[1] >= 2:
        plt.figure(figsize=(6, 6))
        for i in range(signal.shape[1]):
            plt.plot(
                analytic_signal[:, i].real,
                analytic_signal[:, i].imag,
                label=axis_labels[i],
            )
        plt.title("Phase Portrait (Hilbert Plane)")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

        # Phase difference between first two axes
        phase_diff = phase[:, 0] - phase[:, 1]
        plt.figure(figsize=(10, 4))
        plt.plot(
            time, phase_diff, label=f"{axis_labels[0]} - {axis_labels[1]} Phase Diff"
        )
        plt.title("Phase Difference Between First Two Axes")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase Difference (rad)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return envelope, phase, analytic_signal


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(data, lowcut=3, highcut=6, fs=60, order=3):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


def analyze_tremor_with_hilbert(acc, time, fs=60, axis_labels=["X", "Y", "Z"]):
    """
    Analyze IMU tremor features using Hilbert transform on filtered acceleration.

    Parameters:
        acc (ndarray): Raw accelerometer data (shape: [N, 3])
        time (ndarray): Time vector (length N)
        fs (float): Sampling frequency in Hz
        axis_labels (list): Optional axis labels
    Returns:
        envelope, phase, analytic_signal
    """
    # Bandpass filter to isolate tremor band
    filtered = bandpass_filter(acc, fs=fs)

    # Hilbert transform
    analytic_signal = hilbert(filtered, axis=0)
    envelope = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal), axis=0)

    # Plot: Envelope
    plt.figure(figsize=(12, 4))
    for i in range(acc.shape[1]):
        plt.plot(time, envelope[:, i], label=f"Envelope {axis_labels[i]}")
    plt.title("Hilbert Envelope (Tremor Amplitude)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Instantaneous phase
    plt.figure(figsize=(12, 4))
    for i in range(acc.shape[1]):
        plt.plot(time, phase[:, i], label=f"Phase {axis_labels[i]}")
    plt.title("Instantaneous Phase (Hilbert)")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Phase portrait
    plt.figure(figsize=(6, 6))
    for i in range(acc.shape[1]):
        plt.plot(
            analytic_signal[:, i].real,
            analytic_signal[:, i].imag,
            label=f"{axis_labels[i]}",
        )
    plt.title("Phase Portrait (Real vs. Imag)")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Phase difference between X and Y
    if acc.shape[1] >= 2:
        phase_diff = np.unwrap(phase[:, 0] - phase[:, 1])
        plt.figure(figsize=(10, 4))
        plt.plot(time, phase_diff, label="Phase X − Y")
        plt.title("Phase Difference Between X and Y")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase Difference (rad)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return envelope, phase, analytic_signal


def analyze_cwt_spectrogram(
    signal, time, fs=60, max_freq=15, wavelet_width=6, title="CWT Spectrogram"
):
    """
    Perform and plot Continuous Wavelet Transform (CWT) spectrogram using the Morlet wavelet.

    Parameters:
        signal (1D array): The signal to analyze (e.g., a single axis of acceleration).
        time (1D array): Time vector (same length as signal).
        fs (float): Sampling frequency in Hz.
        max_freq (float): Maximum frequency to display.
        wavelet_width (float): Wavelet width parameter (controls frequency resolution).
        title (str): Plot title.

    Returns:
        cwt_matrix (2D array): Complex wavelet coefficients (frequency × time).
        freqs (1D array): Frequencies corresponding to the rows of the CWT matrix.
    """
    time = standardize_time(time)
    min_scale = 1
    max_scale = int(fs / 1.0)  # Approx ~1 Hz
    scales = np.arange(min_scale, max_scale)

    # Compute CWT using Morlet wavelet
    cwt_matrix = cwt(signal, morlet2, scales, w=wavelet_width)

    # Convert scales to approximate frequencies (Morlet: f ≈ fs / (w * scale))
    freqs = fs / (wavelet_width * scales)

    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(
        np.abs(cwt_matrix),
        extent=[time[0], time[-1], freqs[-1], freqs[0]],
        aspect="auto",
        cmap="jet",
        origin="lower",
    )
    plt.colorbar(label="Amplitude")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, max_freq)
    plt.tight_layout()
    plt.show()

    return cwt_matrix, freqs
