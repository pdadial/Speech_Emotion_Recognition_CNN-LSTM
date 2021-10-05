
import numpy as np

def _framing(x, win_len, overlap):
    """

    Divides input signal into overlapping frames of equal length
    Parameters
    ----------
    x : input signal
    win_len : frame length
    overlap : overlapping length
    
    Returns
    -------
    a list of frames
    """
    n_frames = int(np.round((len(x) - overlap)/(win_len - overlap)))
    x_frames = np.zeros((n_frames, win_len))
    for i in range(n_frames):
        start = i * (win_len - overlap)
        end = start + win_len
        x_frames[i] = x[start:end] if len(x[start:end])==win_len else np.append(x[start:end],np.zeros(win_len%len(x[start:end])))
    return x_frames

def _deframing(x, x_vad, win_len, overlap):
    """

    Combines overlapping frames
    Parameters
    ----------
    x : original input signal
    x_vad : list of voice activated frames
    win_len : frame length
    overlap : overlapping length
    
    Returns
    -------
    a signal of length equal to the original input signal
    """
    x_voice = np.zeros((len(x)))
    for i in range(len(x_vad)):
        start = i * (win_len - overlap)
        end = i * (win_len - overlap) + win_len
        x_voice[start:end] = x_vad[i]
    for i in range(len(x_voice)):
        if x_voice[i]: 
            x_voice[i] = x[i]   
    return np.trim_zeros(x_voice)

def _calculate_nrg(x_frames):
    """

    Calculates frame energy
    Parameters
    ----------
    x_frames : list of frames
    
    Returns
    -------
    a list of normalised energies of respective frames
    """
    nrgs = np.diagonal(np.dot(x_frames, x_frames.T))
    log_nrgs = np.log(nrgs)
    norm_nrgs = (2*(log_nrgs-min(log_nrgs))/(max(log_nrgs)-min(log_nrgs)))-1
    return norm_nrgs

def VAD(x, fs, db, nrg_th=0, context=5):
    """

    Performs voice activity detection by comparing frame energy to a threshold
    Parameters
    ----------
    x : original input signal
    fs : sampling rate
    db : Database ID (RAVDESS:1 and SAVEE:2)
    nrg_th: energy threshold
    context: number of neighbouring frames to be used in comparison
    
    Returns
    -------
    a list of voice activated frames; where 1 indicates presence of voice and 0 indicates absence of voice
    """
    percent_th = 0.4 if db==1 else 0.3
    win_len = int(fs * 0.025)
    overlap = int(win_len * 0.25)
    x_frames = _framing(x, win_len, overlap)
    n_frames = int(x_frames.shape[0])
    x_nrgs = _calculate_nrg(x_frames)
    x_vad = np.zeros((n_frames, 1))
    for i in range(n_frames):
        start = max(0, i-context)
        end = min(n_frames, i+context)
        num_nrgs_above_th = np.sum(x_nrgs[start:end,...] > nrg_th)
        total_num_nrgs = end - start + 1
        x_vad[i] = (num_nrgs_above_th/total_num_nrgs) > percent_th
    return _deframing(x, x_vad, win_len, overlap)