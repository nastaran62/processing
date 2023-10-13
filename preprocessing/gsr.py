import numpy as np
from scipy import signal
import neurokit2 as nk

class GsrPreprocessing():
    def __init__(self, data, sampling_rate=128):
        self._sampling_rate = sampling_rate
        self.data = data

    def gsr_noise_cancelation(self):
        self.data = nk.eda_clean(self.data, sampling_rate=self._sampling_rate)


    def gsr_noise_cancelation0(self, low_pass=0.1, high_pass=15):
        '''
        '''
        nyqs = self._sampling_rate * 0.5
        # Removing high frequency noises
        b, a = signal.butter(5, [low_pass / nyqs, high_pass / nyqs], 'bands')
        output = signal.filtfilt(b, a, np.array(self.data, dtype=np.float))

        # Removing rapid transient artifacts
        self.data = signal.medfilt(output, kernel_size=5)

    def baseline_normalization(self, baseline_duration=3, baseline=None, normalization=True):
        if normalization is True:
            if baseline is None:
                baseline = self.data[0:self._sampling_rate*baseline_duration]
            else:
                baseline_duration = 0
            baseline_avg = np.mean(baseline)
            self.data = self.data[self._sampling_rate*baseline_duration:] - baseline_avg
        else:
            self.data = self.data[self._sampling_rate*baseline_duration:]

    def get_data(self):
        return self.data
