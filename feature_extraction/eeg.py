import numpy as np
from sklearn.preprocessing import StandardScaler
from pywt import wavedec

class EegFeatures():
    def __init__(self, data, sampling_rate):
        self.data = data
        self.sampling_rate = sampling_rate


    def get_wavelet_features(self):
        small_cA5, small_cD5, small_cD4, small_cD3, small_cD2, small_cD1 = \
            wavedec(self.data, "db4",
                    mode="periodization",
                    level=5)
        theta_small = small_cD5
        alpha_small = small_cD4
        beta_small = small_cD3
        gamma_small = small_cD2
        bands = [theta_small, alpha_small, beta_small]
        all_features = []
        for band in range(len(bands)):
            power = np.sum(bands[band]**2)
            entropy = np.sum((bands[band]**2)*np.log(bands[band]**2))
            all_features.extend([power, entropy])
        return np.array(all_features)


    def get_total_power_bands(self, method="mean"):
        '''
        calculates alpha, betha, delta, gamma bands
        It uses fft

        :param np.array data: EEG data
        :param int sampling_rate: sampling rete

        :rtype np.array(float, float, float, float, float)
        :return delta, theta, alpha, beta, gamma
        '''
        return self._get_power_bands(self.data.ravel(), method=method)


    def _get_power_bands(self, signal, method="mean"):
        sample_count, = signal.shape
        # Get real amplitudes of FFT (only in postive frequencies)
        fft_values = np.absolute(np.fft.rfft(signal))

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(sample_count, 1.0/self.sampling_rate)

        # Define EEG bands
        eeg_bands = {'Delta': (0, 4),
                    'Theta': (4, 8),
                    'Alpha': (8, 12),
                    'Beta': (12, 30),
                    'Gamma': (30, 45)}
        eeg_band_fft = dict()
        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                            (fft_freq < eeg_bands[band][1]))[0]

            if(fft_values[freq_ix].size == 0):
                eeg_band_fft[band] = 0
            else:
                if method == "mean":
                    eeg_band_fft[band] = np.mean(fft_values[freq_ix])
                elif method == "psd":
                    eeg_band_fft[band] = np.sum(fft_values[freq_ix]**2)
                elif method == "entropy":
                    eeg_band_fft[band] = np.sum((fft_values[freq_ix]**2)*np.log(fft_values[freq_ix]**2))
        bands = np.array([eeg_band_fft['Alpha'],
                        eeg_band_fft['Beta'],
                        eeg_band_fft['Theta'],
                        eeg_band_fft['Delta'],
                        eeg_band_fft['Gamma']])
        return bands


    def get_channels_power_bands(self, method="mean"):
        '''
        calculates alpha, betha, delta, gamma bands for each channel
        It uses fft

        :param np.array data: EEG data
        :param int sampling_rate: sampling rete

        :rtype np.arraye(np.array(float)), shape is channel_count * 5
        :return an array of power bands for all channels
        '''
        columns, samples = self.data.shape
        ch = 0
        eeg_bands = []
        while ch < columns:
            eeg_band = self._get_power_bands(self.data[ch, :], method=method)
            ch += 1
            eeg_bands.append(eeg_band)
        return ((np.array(eeg_bands)).ravel())
