import numpy as np
import heartpy as hp
import neurokit2 as nk

class PpgPreprocessing():
    def __init__(self, data, sampling_rate=128):
        self._sampling_rate = sampling_rate
        self.data = data

    def neurokit_filtering(self):
        self.data = nk.ppg_clean(self.data, sampling_rate=self._sampling_rate, method="nabian2018")
        print(self.data, "neurokit")


    def filtering(self, low_pass=0.5, high_pass=4):
        filtered = hp.filter_signal(self.data,
                                    [low_pass, high_pass],
                                    sample_rate=self._sampling_rate,
                                    order=3,
                                    filtertype='bandpass')

                                    

        self.data = filtered

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
