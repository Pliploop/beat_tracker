# plot a spectrogram

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def show_spectrogram(spec,sr = 22050, fps = 100):
    fig = plt.figure(figsize=(12, 4))
    hop_length = sr // fps
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length, cmap='Greys')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()
    
    return fig
    
    
def show_beats_and_downbeats(beats,downbeats,sr = 22050, fps = 100):
    hop_length = sr // fps
    fig = plt.figure(figsize=(12, 4))
    times = np.arange(0, len(beats)/fps, 1/fps)
    plt.plot(times, beats, label='beats')
    plt.plot(times, downbeats, label='downbeats')
    plt.xlabel('Time (s)')
    plt.ylabel('Beat')
    plt.title('Beats and downbeats')
    plt.legend()
    plt.show()
    
    return fig
    
    
def show_beats_and_spectrogram(spec,beats,downbeats,sr = 22050, fps = 100):
    hop_length = sr // fps
    fig = plt.figure(figsize=(12, 8))
    times = np.arange(0, len(beats)/fps, 1/fps)
    plt.subplot(2,1,1)
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length, cmap='Greys')
    plt.title('Mel spectrogram')
    plt.subplot(2,1,2)
    plt.plot(times, beats, label='beats', color='k')
    plt.plot(times, downbeats, label='downbeats', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Beat')
    plt.title('Beats and downbeats')
    # set xlim to the length of the beats
    plt.xlim(0, len(beats)/fps)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    return fig
    