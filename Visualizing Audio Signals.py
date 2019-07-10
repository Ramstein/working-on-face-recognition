import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Read the audio file
sampling_freq, signal = wavfile.read('sample1.wav')

#Display the params
print('\nSignal shape: ', signal.shape)
print("Datatype: ", signal.dtype)

print("Signal duration: ", round(signal.shape[0] / float(sampling_freq), 8), ' seconds')

#Normalize the signal
signal = signal / np.power(2, 15)

#Extract the first 50 values
signal = signal[ : 50]

#Construct the time axis in milliseconds
time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq)

#Plot the audio signal
#plt.plot(time_axis, signal, color="black")
#plt.xlabel("Time (milliseconds)")
#plt.ylabel("Amplitude")
#plt.title("Input audio signal")
#plt.show()


#extract the length of the audio signal
len_signal = len(signal)

#Extracting the half length of the audio signal
len_half = np.ceil((len_signal + 1) / 2.0).astype(np.int)

#Apply fourier transformer to the signal
freq_signal = np.fft.fft(signal)

# Normalization
freq_signal = abs(freq_signal[0:len_half]) / len_signal
# Take the square
freq_signal **= 2
# Extract the length of the frequency transformed signal
len_fts = len(freq_signal)
# Adjust the signal for even and odd cases
if len_signal % 2:
    freq_signal[1:len_fts] *= 2
else:
    freq_signal[1:len_fts-1] *= 2

# Extract the power value in dB
signal_power = 10 * np.log10(freq_signal)
# Build the X axis
x_axis = np.arange(0, len_half, 1) * (sampling_freq / len_signal) / 1000.0 

# Plot the figure
plt.figure()
plt.plot(x_axis, signal_power, color='black')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Signal power (dB)')
plt.show()
 











