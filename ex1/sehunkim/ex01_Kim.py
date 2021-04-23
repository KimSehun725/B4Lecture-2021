import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
from scipy import signal

#sampling rate = 44100Hz
wave_array, sr = librosa.load('/Users/sehunkim/Library/Mobile Documents/com~apple~CloudDocs/School/2021/TodaLab/Rinko/B4Lecture-2021/ex1/sehunkim/sample.wav')
frame_size = 300                                        # size of one frame of stft
overlap = 70                                            # number of samples to overlap
window_func = signal.hamming(frame_size)                # variable for window function
n_of_frame = len(wave_array) // (frame_size - overlap)  # calculating the number of frames
p_wave_array = np.pad(wave_array,(0,overlap))           # zero padding
time = wave_array / sr                                  # calculating total time

def stft(data_array, frame_size, overlap):
    result = np.array([])
    for frame_no in range (n_of_frame) :
        left = frame_no * (frame_size - overlap)        # left end of frame
        right = left + frame_size                       # right end of frame
        frame = data_array[left : right]                # cutting one frame
        frame = frame * window_func                     # apply window function
        result = np.append(result, fft(frame))          # append the result of fft
    result = result.reshape(n_of_frame,frame_size)
    return result

def istft (data_array, frame_size, overlap):
    result = np.array([])
    for frame_no in range (n_of_frame) :
        result = np.append(result, ifft(data_array[frame_no])[:frame_size - overlap])
    return result

stft_result = stft(p_wave_array, frame_size, overlap)
db = librosa.amplitude_to_db((np.abs(stft_result)).T)   # transpose and convert to db
db = db[:len(db)//2]                                    # cutting out the symetrical part

istft_result = istft(stft_result, frame_size, overlap)
istft_result = istft_result[:(len(wave_array) - frame_size)]

# plot original waveform
t = np.arange(0, len(wave_array)) / sr
plt.figure(figsize = (15,10))
plt.subplots_adjust(wspace = 0.2, hspace=0.6)
plt.subplot(3,1,1)
plt.plot(t, wave_array)
plt.title('Original Signal')
plt.xlabel('Time[s]')
plt.ylabel('Magnitude')

# plot spectogram of the result of stft
plt.subplot(3,1,2)
print(db.shape)
print(wave_array.shape)
librosa.display.specshow(db, y_axis = 'linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectogram')

# plot re-synthesized waveform
p_istft_result = np.pad(istft_result,(0, len(t) - len(istft_result))) # zero padding
plt.subplot(3,1,3)
plt.plot(t, p_istft_result)
plt.title('Re-synthesized Signal')
plt.xlabel('Time[s]')
plt.ylabel('Magnitude')
plt.show()
