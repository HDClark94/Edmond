import numpy as np 

def addNoise(signal, noise_ratio=0.2, noise_segment_ratio=0.2, snr=0.5, Fs=30000):
    # noise_segment_ratio: how long is one noise segment
    # noise_ratio: portion of the orignal signal contaminated by noise
    # snr: signal to noise ratio of the resulting signal
    
    noise_segment_length = int(Fs*noise_segment_ratio)
    start = signal.shape[1]//2
    segment_start = np.arange(start, signal.shape[1]-noise_segment_length, noise_segment_length) 
    noise_segment = np.random.choice(segment_start, int(len(segment_start)*noise_ratio), replace=False)
    signal_var = np.mean(np.var(signal,axis=1)) # calculate the noise level of the original signal
    noise_var = signal_var/snr
    signal_noise = signal.copy()
    for n in noise_segment:
        signal_noise[:,n:n+noise_segment_length] += (np.random.rand(signal.shape[0],noise_segment_length)-0.5)*2*np.sqrt(noise_var)

    return signal_noise
