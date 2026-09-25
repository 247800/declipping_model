import torch
import torchaudio

waveform, sample_rate = torchaudio.load("dataset/jazz.00000.wav")
print(waveform)
print(sample_rate)
print(min(waveform))
print(max(waveform))