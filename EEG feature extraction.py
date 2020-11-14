import pywt
import warnings
import argparse
import pandas as pd 
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from scipy import stats, signal, integrate
from numpy import linalg
from nolds import lyap_r
from lempel_ziv_complexity import lempel_ziv_complexity


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


def import_dataset(file_location, sampling_frequency, n_channels):
    
    with open(file_location) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        
        new_format = []
        classes = []
        
        for row in content:
            
            row = row.split()
            classes.append(row.pop(0))
            
            row = np.array( row ).reshape((n_channels, -1)).astype(float)
            new_format.append(row)
            
        f.close()
        return np.array([new_format, classes])


# Izlucivanje znacajki iz vremenske domene

# In[4]:


def get_time_features(sequence):
    #sequence should be of (channels x values) format
    
    minimal = np.min(sequence, axis = 1)
    maximal = np.max(sequence, axis = 1)
    mean = np.mean(sequence, axis = 1)
    median = np.median(sequence, axis=1)
    std = np.std(sequence, axis = 1)
    skew = sp.stats.skew(sequence, axis = 1)
    kurtosis = sp.stats.kurtosis(sequence, axis = 1)
    
    result = np.array( [minimal, maximal, mean, median, std, skew, kurtosis] )
    
    result = pd.DataFrame(result.transpose(), columns = ["min","max","mean","median","std","skew","kurtosis"])
    result.index.name = "channel"

    return result
    
    


# In[5]:


def get_hjorth_features(sequence):
    #returns hjorth parameters as features, those are activity, mobility and complexity
    
    activity = np.var(sequence, axis = 1)
    
    
    channels = sequence.shape[0]
    n = sequence.shape[1]
    
    D = np.diff(sequence)

    # pad the first difference
    D = np.concatenate((sequence[:,0].reshape(channels,-1), sequence), axis = 1)

    #M2 = float(sum(D ** 2)) / n
    M2 = np.sum(D**2, axis=1) / float(n)
    TP = np.sum(sequence ** 2, axis=1)
    M4 = np.zeros(channels)
    for i in range(1, n):
        M4 += (D[:,i] - D[:,i - 1]) ** 2
    M4 = M4 / n
    
    mobility = np.sqrt(M2/TP)
    complexity = np.sqrt(  M4*TP/M2/M2  )
    
    result = np.array([activity, mobility, complexity]).transpose()
    result = pd.DataFrame(result, columns = ["activity","mobility","complexity"])
    result.index.name = "channel"
    
    return result
    


# Frekvencijska domena

# Koristeći Fourierovu transformaciju signal možemo transformirati u frekvencijsku
# domenu. Prednost frekvencijske domene jest mogućnost izračuna spektralne gustoće
# snage (eng. Power Spectral Density, PSD). Sam izračun PSD je moguće izračunati raz-
# ličitim metodama med̄u kojima su klasični periodogram, Welchova metoda, Burgova
# metoda. Zatim iz PSD računamo same značajke. Možemo izračunati mjeru spektral-
# nog momenta koja se dobiva kao umnozak frekvencije i procijenjene snage te frek-
# vencije. Spektralna entropija je značajka koja nam ukazuje na kaos u frekvencijskoj
# domeni. Maksimalna vrijednost entropije te frekvencija na kojoj se postiže se tako-
# d̄er mogu koristiti kao značajke. Rubna spektralna frekvencija predstavlja frekvenciju
# ispod koje x posto (uobičajno izmed̄u 75 i 95) ukupne snage signala leži. Na PSD mo-
# žemo raditi različite statističke značajke koje je često pametno primijeniti na raspone
# frekvencija koje korespondiraju spomenutim karakterističnim moždanim valovima.

# In[6]:


def get_psd_features(signal, fs):
    #signal, fs - sampling frequency of signal
    #psd using welch method
    
    #delta,theta,alpha,beta,gamma
    brainwaves_ranges = [ 1,4,8,12,25, fs ]
    
    freq, psd = sp.signal.welch(signal, fs)
    
    alpha = psd[:, brainwaves_ranges[2]:brainwaves_ranges[3]]
    beta  = psd[:, brainwaves_ranges[3]:brainwaves_ranges[4]]
    gamma = psd[:, brainwaves_ranges[4]:brainwaves_ranges[5]]
    delta = psd[:, brainwaves_ranges[0]:brainwaves_ranges[1]]
    theta = psd[:, brainwaves_ranges[1]:brainwaves_ranges[2]]
    
    alpha_mean = np.mean(alpha, axis=1)
    beta_mean = np.mean(beta, axis=1)
    gamma_mean = np.mean(gamma, axis=1)
    delta_mean = np.mean(delta, axis=1)
    theta_mean = np.mean(theta, axis=1)
    
    psd_sum = np.sum(psd, axis=1)
    
    alpha_rel = np.sum(alpha, axis=1)/psd_sum 
    beta_rel = np.sum(beta, axis=1)/psd_sum 
    gamma_rel = np.sum(gamma, axis=1)/psd_sum 
    delta_rel = np.sum(delta, axis=1)/psd_sum 
    theta_rel = np.sum(theta, axis=1)/psd_sum 
    
    spm_first_order = np.trapz(psd*freq, freq)
    spm_second_order = np.trapz(psd*np.square(freq), freq)
    
    SP = np.argmax(psd, axis=1)
    POW_SP = np.max(psd, axis=1)
    
    #spectral entropy
    SPE = -np.trapz( psd*np.log(psd) )/np.log(len(freq))
    
    #relativni meanovi
    at_div_b = (alpha_mean+ theta_mean) / beta_mean
    a_div_b = alpha_mean / beta_mean
    at_div_ab = (alpha_mean + theta_mean) / (alpha_mean+beta_mean)
    t_div_b = theta_mean/beta_mean
    gb_div_da = (gamma_mean + beta_mean) / (delta_mean + alpha_mean)
    g_div_d = gamma_mean/delta_mean
    
    result = np.array([   alpha_mean, beta_mean, gamma_mean, delta_mean, theta_mean, 
                       alpha_rel, beta_rel, gamma_rel, delta_rel, theta_rel, 
                      spm_first_order, spm_second_order, SP, POW_SP, SPE,
                       at_div_b, a_div_b, at_div_ab, t_div_b, gb_div_da, g_div_d])
    
    
    #α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ

    result = pd.DataFrame(result.transpose(), columns = ["psd_alpha_mean","psd_beta_mean","psd_gamma_mean",
                                                         "psd_delta_mean", "psd_theta_mean",
                                                         "alpha_rel", "beta_rel", "gamma_rel", "delta_rel",
                                                         "theta_rel", "spm_1", "spm_2", "SP", "POW_SP", "SPE",
                                                        "α+θ/β", "α/β","α+θ/α+β","θ/β","γ+β/δ+α","γ/δ" ])
    result.index.name = "channel"
    
    return result


# Nonlinear features

# Lempel-Zivova kompleksnost je mjera kompleksnosti koja se može koristiti kao
# mjera ponavljanja binarnih sekvenci. Stoga prvo moramo pretvoriti vremenski niz
# u binarni niz. To možemo učiniti usporedbom s nekim pragom. Ako je amplituda veća
# od praga element niza je 1, dok je u suprotnom 0. Često se medijan niza koristi kao
# prag. Zatim skeniramo binarni niz s lijeva na desno te varijablu brojača kompleksnosti
# 12c(n) povečavamo za 1 svaki put kad naid̄emo na novi podniz.

# In[7]:


def get_lempel_ziv_complexity(signal):
    
    threshold = np.median(test_row, axis=1)
    binary = np.greater(np.expand_dims(th, axis=1), test_row)
    binary = binary.astype(int).astype(str)
    lzc = [lempel_ziv_complexity(  "".join(x)  ) for x in binary]
    
    result = np.array([lzc]).transpose()
    result = pd.DataFrame(result, columns = ["lzc"])
    result.index.name = "channel"
    
    return result


# In[8]:


def get_hfd(X, Kmax=8):
    
    result = []
    for channel in X:
        L = []
        x = []
        N = len(channel)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(channel[m + i * k] - channel[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(float(1) / k), 1])

        (p, _, _, _) = np.linalg.lstsq(x, L)
        result.append(p[0])

    result = np.array([result]).transpose()
    result = pd.DataFrame(result, columns = ["hfd"])
    result.index.name = "channel"
    
    return result


# In[9]:


def get_shannon_entropy(signal, K=2,  bin_size=10):
    
    results = []
    
    for channel in signal:
        hist, bins = np.histogram(channel)
        hoho = hist/len(channel)
        result = -K*np.sum(hoho*np.log(hoho))
        results.append(result)
        
    result = np.array([results]).transpose()
    result = pd.DataFrame(result, columns = ["entropy_shan"])
    result.index.name = "channel"
    
    return result
    


# In[10]:


def get_hurst_ernie_chan(signal):
    
    lags = range(2,100)
    results = []
    
    for p in signal:
    
        variancetau = []
        tau = []

        for lag in lags: 

            tau.append(lag)

            pp = np.subtract(p[lag:], p[:-lag])
            variancetau.append(np.var(pp))

        m = np.polyfit(np.log10(tau),np.log10(variancetau),1)

        hurst = m[0] / 2
        results.append(hurst)

    result = np.array([results]).transpose()
    result = pd.DataFrame(result, columns = ["hurst_exp"])
    result.index.name = "channel"
    
    return result


# In[11]:


def get_lyapunov_exponent(signal):
    
    result = []
    
    for channel in signal:
        
        result.append(lyap_r(channel))
        
    result = np.array([result]).transpose()
    result = pd.DataFrame(result, columns = ["lyapunov_exp"])
    result.index.name = "channel"
    
    return result


# In[12]:


def get_wavelet_features(signal, fs, wavelet = "db2"):
    
    #hardcoded level for signals sampled with f = 256Hz
    
    lvl = 7
    
    coeff = pywt.wavedec(signal, wavelet=wavelet, level = lvl)
    
    delta = coeff[0]
    theta = coeff[1]
    alpha = coeff[2]
    beta = coeff[3]
    gamma = np.concatenate(coeff[4:], axis=1) #not sure if correct
    
    alpha_mean = np.mean(alpha, axis=1)
    beta_mean = np.mean(beta, axis=1)
    gamma_mean = np.mean(gamma, axis=1)
    delta_mean = np.mean(delta, axis=1)
    theta_mean = np.mean(theta, axis=1)
    
    results = np.array([alpha_mean, beta_mean, gamma_mean, delta_mean, theta_mean]).transpose()
    results = pd.DataFrame(results, columns = ["wt_alpha_mean", "wt_beta_mean","wt_gamma_mean", 
                                             "wt_delta_mean", "wt_theta_mean"])
    results.index.name = "channel"
    
    return results


# In[13]:


def get_dfa_mois(signal, min_window_size=4, max_window_size=8):
    
    alphas = []
    
    for channel in signal:
        
        alphas.append(dfa_single_channel(channel,min_window_size, max_window_size))
        
    result = np.array([alphas]).transpose()
    result = pd.DataFrame(result, columns = ["DFA"])
    result.index.name = "channel"
    
    return result
        

def dfa_single_channel(a, min_window_size, max_window_size):
    Fns = []
    yk = np.cumsum(a-np.mean(a))
    N  =len(a)
    
    for window_size in range(min_window_size, max_window_size):
        
        nn = N//window_size
        ynk_all = np.array([])

    
        for i in range(nn):

            segment = yk[i*window_size:(i+1)*window_size]
            m, c = np.polyfit( np.arange(window_size), segment, 1 )
            ynk = m*np.arange(window_size)+c
            ynk_all = np.append( ynk_all, ynk)
        
        last_segment = N%window_size
        yk_temp = yk
        if last_segment > 2:
            last_segment = yk[-last_segment:]
            m, c = np.polyfit( np.arange(len(last_segment)), last_segment, 1 )
            ynk = m*np.arange(len(last_segment))+c
            ynk_all = np.append( ynk_all, ynk)
            
        elif last_segment !=0:
            yk_temp = yk[:-last_segment]

        Fn = np.sqrt(np.sum(np.square(yk_temp - ynk_all)))
        Fns.append(Fn)
        
    L = np.arange(min_window_size,max_window_size)

    Alpha = np.linalg.lstsq(np.vstack(
        [np.log(L), np.ones(len(L))]
    ).T, np.log(Fns))[0][0]    
    
    return Alpha


# In[14]:


def pearson(a,b):
        
    a_diff = a-np.mean(a)
    b_diff = b-np.mean(b)    
    r = np.sum(a_diff*b_diff)/(   np.sqrt(np.sum(np.square(a_diff))*np.sum(np.square(b_diff)))   )
    
    return r

def get_pearson_features(x):
    
    df = pd.DataFrame(x)
    series =[]
    for row in range(0, df.shape[0]):
        series.append(df.corrwith(df.iloc[row], axis = 1))

    result = pd.DataFrame(series)
    result.columns = ["PCC ch-" + str(colname) for colname in result.columns]
    result.index.name = "channel"
    return result


# In[15]:


def get_all_features(x, sampling_frequency):
    
    time_features = get_time_features(x)
    hjorth_features = get_hjorth_features(x)
    psd_features = get_psd_features(x, sampling_frequency)

    hurst_exponent = get_hurst_ernie_chan(x)
    higuchi = get_hfd(x)
    lypunov_exponent = get_lyapunov_exponent(x)
    wavelet_features = get_wavelet_features(x, sampling_frequency)
    dfa = get_dfa_mois(x)
    pearson = get_pearson_features(x)
    
    
    master_features = time_features.join( [hjorth_features, psd_features, hurst_exponent,
                                           higuchi, lypunov_exponent, wavelet_features, dfa, pearson] )


    return master_features
    


# In[16]:


def input_data_to_features(X, sampling_frequency):
    
    master_df = get_all_features(X[0], sampling_frequency)
    master_df = master_df.reset_index()
    master_df["instance"] = 0
    master_df = master_df.set_index(["instance", "channel"])
    
    n = len(X)
    for instance in range(1, n):
        
        x = X[instance]
        x_features = get_all_features(x, sampling_frequency)
        
        x_features = x_features.reset_index()
        x_features["instance"] = instance
        x_features = x_features.set_index(["instance", "channel"])
        
        master_df = master_df.append(x_features)

    return master_df
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fs', help='sample frequency',
                    default=256, type=int)
    parser.add_argument('--ch', help='number of channels',
                    default=6, type=int)
    parser.add_argument('--i', help='raw signal input file',
                        default="./example/train_data.txt")
    parser.add_argument('--o', help='raw signal input file',
                        default="./example/features.csv")    

    args = parser.parse_args()

    X, Y = import_dataset(args.i, args.fs, args.ch)

    features = input_data_to_features(X, args.fs)
    features.to_csv(args.o)
