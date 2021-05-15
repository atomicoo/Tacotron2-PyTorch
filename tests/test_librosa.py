import librosa
import numpy as np
import soundfile

fullpath = '../audios/LJ001-0007.wav'
sampling_rate = 22050

y, sr = librosa.core.load(fullpath, sampling_rate)
print(y.shape)

print('np.mean(y): ' + str(np.mean(y)))
print('np.max(y): ' + str(np.max(y)))
print('np.min(y): ' + str(np.min(y)))

soundfile.write('test.wav', y, sampling_rate)

y, sr = librosa.core.load('test.wav')
# Trim the beginning and ending silence
yt, index = librosa.effects.trim(y)
print(index)
# Print the durations
print(librosa.core.get_duration(y), librosa.core.get_duration(yt))
print(len(y), len(yt))
soundfile.write('test2.wav', y, sampling_rate)
