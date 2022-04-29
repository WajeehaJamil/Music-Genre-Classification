import numpy as np
from typing import Optional
from librosa.filters import mel
from librosa.core import load as lb_load, stft
from utils import read_yaml
import librosa as lb

cfg = read_yaml()

def extract_mel_band_energies(audio_file: np.ndarray,
                              sr: Optional[int] = cfg['feature_extract']['sr'],
                              n_fft: Optional[int] = cfg['feature_extract']['n_fft'],
                              hop_length: Optional[int] = cfg['feature_extract']['hop_length'],
                              frames: Optional[int] = cfg['feature_extract']['frames'],
                              n_mels: Optional[int] = cfg['feature_extract']['n_mels'])\
                            -> np.ndarray:
                              
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 512.
    :type hop_length: Optional[int]
    :param n_mels: Number of MEL frequencies/filters to be used, defaults to 40.
    :type n_mels: Optional[int]
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    
    spec = stft(
        y=audio_file,
        n_fft=n_fft,
        hop_length=hop_length)
        
    #mel_filters = mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    #spec = np.dot(mel_filters, np.abs(spec) ** 2)

    wave = np.abs(spec) ** 2
    spectrogram = lb.feature.melspectrogram(S=wave, n_mels=n_mels)  
    #spectrogram = lb.power_to_db(spectrogram, ref=np.max)
    spectrogram = lb.amplitude_to_db(spectrogram, ref=np.max)
    norm_spectrogram = spectrogram - np.amin(spectrogram)
    norm_spectrogram = norm_spectrogram / float(np.amax(norm_spectrogram))
     
    return norm_spectrogram[:, np.r_[0:200]]  # final_shape = 40*500
  
  
def method2():
    wave = np.abs(wave) ** 2
    spectrogram = lb.feature.melspectrogram(S=wave, n_mels=N_MELS)  # mel bands (40)
    spectrogram = lb.power_to_db(spectrogram, ref=np.max)
    norm_spectrogram = spectrogram - np.amin(spectrogram)
    norm_spectrogram = norm_spectrogram / float(np.amax(norm_spectrogram))

    if int(norm_spectrogram.shape[1]) < FRAMES:  # 10 sec samples gives 500 frames
      z_pad = np.zeros((N_MELS, FRAMES))
      z_pad[:, :-(FRAMES - norm_spectrogram.shape[1])] = norm_spectrogram
      spec = feature_file.append(z_pad)
    else:
        img = norm_spectrogram[:, np.r_[0:FRAMES]]  # final_shape = 40*500
        spec = img



    feature_file = np.array(feature_file)
    feature_file = np.reshape(feature_file, (len(data_id), N_MELS, FRAMES, 1))
