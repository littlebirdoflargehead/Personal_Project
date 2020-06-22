import numpy as np
import math

from utils import active_vox_generator


def basic_signal():
    '''
    Generate Basic Signal for each source
    :return:
    '''
    # Time
    Time = np.linspace(-1, 1, 501)
    StimTime = np.argmin(np.abs(Time))
    Activetime = np.arange(StimTime, Time.shape[0])

    # Basic Signal
    f = 5
    pi = math.pi
    tao = np.array([0.23, 0.3, 0.48, 0.55]) * np.max(Time[Activetime])
    omega = np.array([0.06, 0.08, 0.07, 0.08]) * np.max(Time[Activetime])
    Basis = np.zeros([4, Time.shape[0]])
    for k in range(4):
        Basis[k, Activetime] = np.sin(2 * pi * f * Time[Activetime]) * np.exp(-np.power(
            ((Time[Activetime] - tao[k]) / omega[k]), 2))
    Basis = Basis / np.sqrt(np.sum(np.power(Basis, 2), axis=1)).reshape(-1, 1)

    return Basis, StimTime


def simulated_source(nSource, Patch, Basis):
    '''
    Generate Simulated Source Signal without noise based on Basic Signal and Patch
    :param nSource: Source number for Cortex
    :param Patch: Number of source(patch)
    :param Basis: Basic Signal for each source
    :return:
    '''
    s_real = np.zeros([nSource, Basis.shape[1]])

    k = Basis.shape[0]
    A = 5 * 1e-9
    AA = A * np.eye(k)
    # AA = A * np.array([[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[2,-2,1,0]])

    for i in range(len(Patch)):
        s_real[Patch[i], :] = np.matmul(AA[np.random.permutation(k)[0], :], Basis)

    return s_real


def awgn(sig, snr=5):
    '''
    Add white Gaussian noise to a signal
    :param sig: signal
    :param snr: Signal Noise Ratio
    :return: signal added noise
    '''
    # measure signal power
    sigPower = np.sum(np.power(sig, 2)) / np.prod(sig.shape)
    reqSNR = 10 ** (snr / 10)
    noisePower = sigPower / reqSNR
    noise = math.sqrt(noisePower) * np.random.randn(sig.shape[0], sig.shape[1])

    return sig + noise


def signal_whiten(sig, stim_time):
    '''
    Evaluate the noise whiten matrix for the channel based on the unstimulated signal
    :param sig: sensor signal with noise
    :param stim_time: stimulus time
    :return: noise whiten matrix
    '''
    sig_pre = sig[:, :stim_time]
    sig_pre_mean = np.mean(sig_pre, axis=1).reshape(-1, 1)
    covariance = np.matmul(sig_pre - sig_pre_mean, (sig_pre - sig_pre_mean).T) / (stim_time - 1)
    value, vector = np.linalg.eig(covariance)

    rank = np.linalg.matrix_rank(covariance)
    if rank < covariance.shape[0]:
        ind = np.argsort(value)[::-1]
        value = 1 / value
        value[ind[rank:]] = 0.
        W = (np.sqrt(value) * vector).T
        W = W[ind[:rank], :]
    else:
        W = (np.sqrt(1 / value) * vector).T

    return W


def tbf_svd(signal, k=None):
    '''
    Evaluate the temporal basis functions with the SVD method
    :param signal: sensor signal with noise
    :param k: number of temporal basis functions
    :return: Estimated temporal basis functions
    '''
    nSensor = signal.shape[0]
    u, s, vh = np.linalg.svd(signal)
    if k is None:
        s_pow = np.power(s, 2)
        index = np.argwhere((s_pow / np.sum(s_pow)) > (1 / nSensor)).reshape(-1)
        TBFs = vh[index, :]
        TBFs_svd = s[index]
    else:
        TBFs = vh[0:int(k), :]
        TBFs_svd = s[0:int(k)]

    return TBFs, TBFs_svd


def simulated_signal_generator(nPatches=1, extent=6e-4, nTBFs=5, basic=None, 
                               stim=251, snr=5, gain=None, cortex=None):
    '''
    Generate simulated M/EEG signal with white gaussian noise
    :param nPatches: number of patches/source
    :param extent: extent of each source
    :param nTBFs: number of temporal basic functions
    :param basic: temporal basic functions
    :param stim: stimulated time
    :param snr: signal noise ratio
    :param gain: gain matrix
    :param cortex: cortex struture
    :return:
    '''
    if basic is None:
        print('lacked of Basic Signal for simulate signal generation')
        basic, stim = basic_signal()
    if cortex is None or gain is None:
        raise ValueError('lacked of Cortex struture and Gain matrix')

    nSensor, nSource = gain.shape

    # 通过随机生成的种子voxel，扩展成对应的patch
    seedvox = np.random.permutation(nSource)[0:nPatches]
    extents = extent * np.ones_like(seedvox)
    Patch, ActiveVox, Area = active_vox_generator(seedvox, extents, cortex)

    # 使用patch与激活voxel生成源空间的模拟信号
    s_real = simulated_source(nSource, Patch, basic)

    # add white gaussian noise to the measure signal and normalize
    B = awgn(np.matmul(gain, s_real), snr=snr)
    ratio = np.max(np.abs(B))
    B = B / ratio

    # Whiten the lead field matrix and the measure signal
    W = signal_whiten(B, stim)
    B_whiten = np.matmul(W, B)

    # extract TBFs from the measure signal with SVD
    TBFs, TBFs_svd = tbf_svd(B, nTBFs)
    TBFs_whiten, TBFs_svd_whiten = tbf_svd(B_whiten, nTBFs)
    
    return ActiveVox, s_real, ratio, B, W, TBFs, TBFs_svd, TBFs_whiten, TBFs_svd_whiten
