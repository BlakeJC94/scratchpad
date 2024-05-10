import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, bessel, filtfilt, ShortTimeFFT, windows
import pywt
from scipy.fftpack import rfft, irfft, fftfreq, fft

# Generate example noisy signal
np.random.seed(0)
fs = 256  # Sampling frequency
T = 2  # Signal length in seconds
f0 = 60.0  # Frequency to be removed from signal
order = 2

# Create signal
N = T* fs
t = np.arange(0, T, 1/fs)
signal = np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 60 * t)  # Example signal with 60Hz noise
noise = 0.3 * np.random.randn(len(t))
noisy_signal = signal + noise

# Design notch filter
b, a = butter(order, [f0 - 5, f0 + 5], "bandstop", output="ba", fs=fs)

# Apply notch filter to noisy signal
filtered_signal_notch = filtfilt(b, a, noisy_signal)

# Design notch filter
b, a = bessel(order, [f0 - 5, f0 + 5], "bandstop", output="ba", fs=fs)

# Apply notch filter to noisy signal
filtered_signal_notch_bessel = filtfilt(b, a, noisy_signal)

# # Wavelet decomposition
# wavelet = 'db4'  # Choose a wavelet
# level = 4  # Choose the number of decomposition levels
# coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
# # Thresholding - Remove coefficients at 60Hz band
# threshold = 0.001
# coeffs[-1] = pywt.threshold(coeffs[-1], threshold, mode='soft')
# # Wavelet reconstruction
# denoised_signal_wavelet = pywt.waverec(coeffs, wavelet)

from statsmodels.robust import mad

wavelet="db4"
level=1
# calculate the wavelet coefficients
coeff = pywt.wavedec( noisy_signal, wavelet, mode="per" )
# calculate a threshold
sigma = mad( coeff[-level] )
# changing this threshold also changes the behavior,
# but I have not played with this very much
uthresh = sigma * np.sqrt( 2*np.log( len( noisy_signal ) ) )
for i in range(5,len(coeff)):
    coeff[i] =  pywt.threshold( coeff[i], value=uthresh, mode="soft" )
# reconstruct the signal using the thresholded coefficients
denoised_signal_wavelet  = pywt.waverec( coeff, wavelet, mode="per" )

# Plot the original and denoised signals
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.subplot(3, 1, 2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.plot(t, filtered_signal_notch, label='Notch Filtered Signal')
plt.plot(t, filtered_signal_notch_bessel, label='Notch Filtered Signal (Bessel)')
plt.subplot(3, 1, 3)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.plot(t, denoised_signal_wavelet, label='Wavelet Denoised Signal')
plt.legend()

plt.tight_layout()
plt.show()


# Compute spectrograms for the noisy signals
g_std = 8
w = windows.gaussian(fs//4, std=g_std, sym=True)
stft = ShortTimeFFT(w, hop=fs//8, fs=fs, scale_to="magnitude")
Sx = stft.stft(noisy_signal)
Sx_notch = stft.stft(filtered_signal_notch)
Sx_notch_bessel = stft.stft(filtered_signal_notch_bessel)
Sx_wavelet = stft.stft(denoised_signal_wavelet)

# Plot spectrograms
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title('Noisy signal')
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]')
plt.imshow(10 * np.log10(np.abs(Sx)), aspect='auto', cmap='viridis', origin='lower', extent=stft.extent(N))
plt.colorbar(label='PSD [dB]')

plt.subplot(2, 2, 2)
plt.title('Notch signal')
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]')
plt.imshow(10 * np.log10(np.abs(Sx_notch)), aspect='auto', cmap='viridis', origin='lower', extent=stft.extent(N))
plt.colorbar(label='PSD [dB]')

plt.subplot(2, 2, 3)
plt.title('Notch Bessel signal')
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]')
plt.imshow(10 * np.log10(np.abs(Sx_notch_bessel)), aspect='auto', cmap='viridis', origin='lower', extent=stft.extent(N))
plt.colorbar(label='PSD [dB]')

plt.subplot(2, 2, 4)
plt.title('Wavelet signal')
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]')
plt.imshow(10 * np.log10(np.abs(Sx_wavelet)), aspect='auto', cmap='viridis', origin='lower', extent=stft.extent(N))
plt.colorbar(label='PSD [dB]')

plt.tight_layout()
plt.show()
