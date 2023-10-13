import numpy as np
import mne
#from autoreject import AutoReject, get_rejection_threshold


class EegPreprocessing():
    def __init__(self, data, channel_names=None, sampling_rate=128):
        shape_0, shape_1 = data.shape
        if shape_0 > shape_1:
            data = np.transpose(data)
        if channel_names is None:
            self._channel_names = \
                ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3",
                 "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]
        else:
            self._channel_names = channel_names

        channel_data = data#/1e6
        channel_types = ["eeg"]*len(self._channel_names)

        self.__info = mne.create_info(self._channel_names,
                                      sampling_rate,
                                      channel_types)
        self.__info['bads'] = []
        self._mne_raw = mne.io.RawArray(channel_data, self.__info)
        self._mne_raw.set_montage('standard_1020')
        self._mne_raw.interpolate_bads(reset_bads=False)
        self._mne_raw.load_data()
        self._sampling_rate = sampling_rate

    def get_data(self):
        return self._mne_raw.get_data()

    def epoching(self):
        events = mne.make_fixed_length_events(self._mne_raw)
        return mne.Epochs(self._mne_raw, events, preload=True)

    def filter_data(self, low_frequency=1, high_frequency=45, notch_frequencies=[50]):
        '''
        Apply notch filter, low pass and high pass (bandpass) filter on mne data
        '''
        print("Apply filtering")
        # Band pass filter
        if high_frequency is not None and high_frequency is not None:
            self._mne_raw.filter(l_freq=low_frequency, h_freq=high_frequency)
        # Notch filter
        if notch_frequencies not in (None, []):
            self._mne_raw.notch_filter(notch_frequencies)

    def rereferencing(self, referencing_value=None):
        '''
        Apply re-referencing on data
        '''
        if referencing_value is None:
            # Prevent mne from adding a default eeg referencing
            self._mne_raw.set_eeg_reference([])
        elif referencing_value == 'average':
            # Average reference. This is normally added by default, but can
            # be added explicitly.
            self._mne_raw.set_eeg_reference('average', projection=True)

        else:
            # Re-reference from an average reference to the mean of channels
            # Example: raw.set_eeg_reference(['Fp1', 'Fp2'])
            self._mne_raw.set_eeg_reference(referencing_value)
        self._mne_raw.apply_proj()

    def interpolate_bad_channels(self, bad_channels=None):
        '''
        Bad channel removal (interpolate bad channels)
        '''
        if bad_channels is None:
            self._mne_raw.load_data()
            self.display()
            bad_channels = []
            
            while True:
                bad_channel = input()
                print("input", bad_channel)
                if bad_channel == "q" or bad_channel == "Q":
                    break
                bad_channels.append(bad_channel)
            print(bad_channels)
        
        
        self._mne_raw.info['bads'] = bad_channels
        self._mne_raw.interpolate_bads(reset_bads=False)

    def ica_component_rejection(self):
        '''
        It uses ICA for removing artifacts, we should manually select the bad
        components. Usually the first components are related to eye artifacts.
        Source: https://cbrnr.github.io/2018/01/29/removing-eog-ica/
        '''
        print("apply ICA")
        ica = mne.preprocessing.ICA(method="extended-infomax", random_state=42)
        #reject = get_rejection_threshold(self.epoching(), ch_types='eeg')
        print("*********************************")
        #print(reject)

        #ica.fit(self._mne_raw, reject=reject, tstep=1)
        ica.plot_properties(self._mne_raw, picks=[0, 1, 2, 3, 4, 5])
        print("write the number of ICA you want to reject. For example for ica001 write 1.")
        rejection_string = \
            input("Enter the list of components that you want to reject separated \
                   by space: ")
        rejection_list = []
        if rejection_string not in (None, [], "", [""]):
            rejection_string_list = rejection_string.split()
            for item in rejection_string_list:
                rejection_list.append(int(item))
        print(rejection_list)
        if len(rejection_list) > 0:
            ica.exclude = rejection_list
            ica.apply(self._mne_raw)

    def reject_bad_epochs(self):
        picks = mne.pick_types(self.__info,
                               meg=False,
                               eeg=True)
        n_interpolates = np.array([1, 4, 32])
        consensus_percs = np.linspace(0, 1.0, 11)
        ar = AutoReject(n_interpolates, consensus_percs, picks=picks)
        cleaned_epochs, reject_log = ar.fit_transform(self.epoching(), return_log=True)
        return cleaned_epochs, reject_log.bad_epochs

    def display(self):
        # self._mne_raw.load_data()
        self._mne_raw.load_data()
        self._mne_raw.plot(duration=3, scalings='auto')
        self._mne_raw.plot_psd()

    def channel_wise_baseline_normalization(self, baseline=None, baseline_duration=0):
        data = self._mne_raw.get_data() 
        if baseline is None:
            baseline = data[:, 0:self._sampling_rate*baseline_duration]
        else:
            baseline_duration = 0   
        baseline_avg = np.mean(baseline, axis=1)
        final_data = \
            data[:, self._sampling_rate*baseline_duration:] - baseline_avg[:, np.newaxis]
        self._mne_raw = mne.io.RawArray(final_data, self.__info)
        self._mne_raw.load_data()

    def normal_baseline_normalization(self, baseline=None, baseline_duration=0):
        data = self._mne_raw.get_data()
        if baseline is None:
            baseline = data[:, 0:self._sampling_rate*baseline_duration]
        else:
            baseline_duration = 0
        baseline_avg = np.mean(baseline)
        final_data = data[:, self._sampling_rate*baseline_duration:] - baseline_avg
        self._mne_raw = mne.io.RawArray(final_data, self.__info)
        self._mne_raw.load_data()
