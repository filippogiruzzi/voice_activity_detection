import numpy as np
import librosa


def extract_features(signal, freq=16000, n_mfcc=5, size=512, step=16):
    # Mel Frequency Cepstral Coefficents
    mfcc = librosa.feature.mfcc(y=signal, sr=freq, n_mfcc=n_mfcc, n_fft=size, hop_length=step)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Root Mean Square Energy
    mel_spectogram = librosa.feature.melspectrogram(y=signal, sr=freq, n_fft=size, hop_length=step)
    rmse = librosa.feature.rms(S=mel_spectogram, frame_length=size, hop_length=step)

    mfcc = np.asarray(mfcc)
    mfcc_delta = np.asarray(mfcc_delta)
    mfcc_delta2 = np.asarray(mfcc_delta2)
    rmse = np.asarray(rmse)

    features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis=0)
    return features
