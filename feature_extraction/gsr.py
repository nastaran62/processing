import numpy as np
import scipy
import neurokit2 as nk
import matplotlib.pyplot as plt

class GsrFeatureExtraction():
    def __init__(self, data, sampling_rate):
        self.data = data
        self.sampling_rate = sampling_rate
    

    def get_feature_vector(self):
        features = (self.mean() + self.median() + self.maximum() + self.minimum() +
                self.variance() + self.standard_deviation() + self.skewness() +
                self.kurtosis() + self.sum_of_positive_derivative() +
                self.sum_of_negative_derivative() + self.get_frequency_peak() +
                self.get_moment(moment=3) + self.get_moment(moment=4) +
                self.get_moment(moment=5) + self.get_moment(moment=6))
        return np.array(features)

    def prop_neg_derivatives(self):
        feature = (self.data < 0).sum()/np.product(self.data.shape)
        return [feature]


    def get_local_maxima(self):
        '''
        Reterns local maximums
        '''
        return [self.data[i] for i in scipy.signal.argrelextrema(self.data, np.greater)[0]]


    def get_local_minima(self):
        '''
        Reterns local minimums
        '''
        return [self.data[i] for i in scipy.signal.argrelextrema(self.data, np.less)[0]]


    def get_frequency_peak(self):
        '''
        Reterns frequency of occurrence of local extremes
        '''
        local_maxima = self.get_local_maxima()
        local_minima = self.get_local_minima()

        freq_extremes = len(local_maxima) + len(local_minima)

        return [freq_extremes]


    def get_max_amp_peak(self):
        '''
        Returns the highest value of the determined maximums if it exists. Otherwise it returns zero
        '''
        local_maxima = list(self.get_local_maxima()) + [0]
        return [max(local_maxima)]


    def get_var_amp_peak(self):
        '''
        Returns variance of amplitude values calculated for local extremes
        '''
        amplitude_of_local_maxima = np.absolute(self.get_local_maxima())
        amplitude_of_local_minima = np.absolute(self.get_local_minima())
        if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
            return [0]
        variance = np.var(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

        return [variance]


    def std_amp_peak(self):
        '''
        Returns the standard deviation calculated for local extremes
        '''

        local_extremes = list(self.get_local_maxima()) + \
                         list(self.get_local_minima())
        if len(local_extremes) == 0:
            return [0]
        return [np.std(local_extremes)]


    def skewness_amp_peak(self):
        '''
        Retruns the skewness calculated for amplitude of local extremes
        '''
        amplitude_of_local_maxima = np.absolute(self.get_local_maxima())
        amplitude_of_local_minima = np.absolute(self.get_local_minima())
        if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
            return [0]
        skewness = scipy.stats.skew(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

        return [skewness]


    def kurtosis_amp_peak(self):
        '''
        Retruns the kurtosis calculated for amplitude of local extremes
        '''
        amplitude_of_local_maxima = np.absolute(self.get_local_maxima())
        amplitude_of_local_minima = np.absolute(self.get_local_minima())
        if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
            return [0]
        kurtosis = scipy.stats.kurtosis(list(amplitude_of_local_maxima) +
                                        list(amplitude_of_local_minima))

        return [kurtosis]


    def max_abs_amp_peak(self):
        '''
        Retruns the kurtosis calculated for amplitude of local extremes
        '''
        amplitude_of_local_maxima = np.absolute(self.get_local_maxima())
        amplitude_of_local_minima = np.absolute(self.get_local_minima())
        if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
            return [0]
        max_val = max(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

        return [max_val]


    def variance(self):
        '''
        Returns the variance of the data
        '''
        var = np.var(self.data)

        return [var]


    def standard_deviation(self):
        '''
        Returns the standard deviation of the data
        '''
        std = np.std(self.data)

        return [std]


    def skewness(self):
        '''
        Returns the skewness of the data
        '''
        skewness = scipy.stats.skew(self.data)

        return [skewness]


    def kurtosis(self):
        '''
        Returns kurtosis calculated from the data
        '''
        kurtosis = scipy.stats.kurtosis(self.data)

        return [kurtosis]


    def sum_of_positive_derivative(self):
        '''
        Retruns the sum of positive values of the first derivative of the data
        '''
        first_derivative = np.diff(self.data, n=1)
        pos_sum = sum(d for d in first_derivative if d > 0)

        return [pos_sum]


    def sum_of_negative_derivative(self):
        '''
        Returns the sum of the negative values of the first derivative of the data
        '''
        first_derivative = np.diff(self.data, n=1)
        neg_sum = sum(d for d in first_derivative if d < 0)

        return [neg_sum]


    def mean(self):
        '''
        Returns the mean of the data
        '''
        mean = np.mean(self.data)

        return [mean]


    def median(self):
        '''
        Returns the median of the data
        '''
        median = np.median(self.data)

        return [median]


    def range(self):
        '''
        Retruns the range of the data
        '''
        range = max(self.data) - min(self.data)

        return [range]


    def maximum(self):
        '''
        Returns the maximum of the data
        '''
        return [max(self.data)]


    def minimum(self):
        '''
        Returns the minimum of the data
        '''
        return [min(self.data)]

    def get_frequencies(self):
        freq_data = np.abs(np.fft.fftshift(self.data))
        features = (np.mean(freq_data) + np.median(freq_data) +
                    np.std(freq_data) +
                    np.max(freq_data) + np.min(freq_data) + (np.max(freq_data)-np.min(freq_data)))

        return features
    
    def get_moment(self, moment):
        # 3, 4, 5, 6 moment (https://dergipark.org.tr/en/download/article-file/272082)
        return [scipy.stats.moment(self.data, moment=moment)]
    

    

