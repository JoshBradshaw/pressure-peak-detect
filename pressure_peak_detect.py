from __future__ import division
import collections
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, lfilter

"""
Algorithm adapted from:
An Open-source Algorithm to Detect Onset of Arterial Blood Pressure Pulses 
by: W Zong, T Heldt, GB Moody, RG Mark 
"""
### USER TUNABLE FILTER SETTINGS
SAMPLING_RATE = 62 # Hz !!! MUST BE SET CORRECTLY
LOWPASS_FILTER_CUTTOFF_FREQUENCY = 14 # Hz -- must be << SAMPLING_RATE / 2
LOWPASS_FILTER_ORDER = 2 # 2 is recommended by Zong et. al as a starting point

### USER TUNABLE WAVEFORM TRANSFORM SETTINGS
# 128 ms window recommended by Zong et. Al for Adult BP. It is sometimes 
# necessary to shorten this window length when dealing with very fast heartrates
SSF_WINDOW_LENGTH = int(math.floor(SAMPLING_RATE * 0.128))

### USER TUNABLE PEAK THRESHOLDING SETTINGS
# BPM -- used to calculate the refractory period
# you must use a much higher value is the subject has arrythmia or skip-beats
MAX_EXPECTED_HR = 320 # BPM
REFRACTORY_PERIOD = int(math.floor((60 / MAX_EXPECTED_HR) * SAMPLING_RATE))

# the peak threshold is initialized as ~2-4 times the mean SSF signal value,
# then dyanamically updated as peaks are detected
INITIAL_PEAK_THRESHOLD = 3 # arbitrary coefficient -- should be 1-5

# this value determines how many seconds of recording the initial ssf peak
# threshold is based on. Typically 10 seconds works well, but may be too
# long for some very short recordings
INIT_PEAK_THRESHOLD_TIME = 10 # seconds

# to avoid false positive detections of peaks, only detect a peaks
# that exceed this fraction of the magnitude of the last 5 detected peaks 
PEAK_THRESHOLD_FRACTION = 0.75 # 0-1

# number of peaks to average the peak threshold value over. A higher number
# will decrease the risk of false positive detections, but increase the chance
# of false negative (misses) detections in the event of a sudden baseline change
PEAK_BUFFER_LEN = 5 # number of peaks

# ms -- a global trigger shift to compensate for the fact that pressure
# pulses lag the R-waves of the ECG signal. The R-wave location is a more 
# desirable gating window beginning point than the pressure peak location
# for most MRI reconstruction applications
R_WAVE_DELAY_IN_SAMPLES = 8
print R_WAVE_DELAY_IN_SAMPLES

# the stability of the rolling point peak detection algorithm is improved
# marginally if a larger rolling point spacing is used.
ROLLING_POINT_SPACING = 4 # samples

def butter_lowpass(highcut, fs, order=5):
    """generates the filter coefficients for a butterworth lowpass filter,
    see: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html
    """
    nyq = 0.5 * fs # nyquist rate in Hz
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, highcut, fs, order=5):
    """
    applies the butterworth lowpass filter in the conventional time-forward manner
    note that this filtering method will result in a positive phase (time) delay
    on the output, which may need to be accounted for in the R_WAVE_DELAY setting
    """
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def ssf(x0, x1):
    """slope sum function"""
    if x1 > x0:
        return x1 - x0
    else:
        return 0

def windowed_slope_sum_function(fitered_bp_vector, window_length):
    """see the mathematical description of this algorithm in:
    An Open-source Algorithm to Detect Onset of Arterial Blood Pressure Pulses    
    """
    # this could be replaced with simple pointer manipulation, but this is easier
    window_ring_buffer = collections.deque(np.zeros(window_length), maxlen=window_length)

    ssf_vector = []
    window_ssf = 0
    for bp_sample in fitered_bp_vector:
        window_ring_buffer.append(bp_sample)
        window_ssf += ssf(window_ring_buffer[-1], window_ring_buffer[-2])
        window_ssf -= ssf(window_ring_buffer[1], window_ring_buffer[0])
        ssf_vector.append(window_ssf)
    return ssf_vector
    
class PeakThreshold(object):    
    """
    the peak threshold value is simply a rolling average of the the preceding
    peak values.      
    """    
    def __init__(self, ssf_vector, sampling_rate):
        samples_to_avg = INIT_PEAK_THRESHOLD_TIME * sampling_rate
        # takes the mean of the first INIT_PEAK_THRESHOLD_TIME seconds of the SSF waveform
        # assumes that the SSF waveform is very bottom heavy ie. mean << max
        initial_threshold = INITIAL_PEAK_THRESHOLD * np.mean(ssf_vector[:samples_to_avg])
        init_peak_buffer = [initial_threshold for _ in xrange(PEAK_BUFFER_LEN)]
        self.threshold_sum = PEAK_BUFFER_LEN * initial_threshold
        self.peak_buffer = collections.deque(init_peak_buffer, maxlen=PEAK_BUFFER_LEN)  
        
    def get_threshold(self):
        return (self.threshold_sum / PEAK_BUFFER_LEN) * PEAK_THRESHOLD_FRACTION
    
    def update(self, new_peak_value):
        self.threshold_sum -= self.peak_buffer[0]
        self.peak_buffer.append(new_peak_value)
        self.threshold_sum += new_peak_value
        return self.get_threshold()
        
def test_peak_threshold():
    pt = PeakThreshold(np.ones(10000), 250)
    print pt.get_threshold()
    print pt.update(5)
    print pt.update(5)

def estimate_all_R_wave_locations(bp_vector, sampling_rate, output_type='sample_idx'):    
    """returns a list of the R wave locations, in samples
    
    output_types:
        sample_idx -- a list of the bp_vector sample indexes which correspond to R-wave locations
        time -- the times of the R wave locations in seconds, relative to the start of bp_vector
    """
    R_wave_locations = []  

    lp_filtered_vector = butter_lowpass_filter(bp_vector, LOWPASS_FILTER_CUTTOFF_FREQUENCY, 
                              sampling_rate, order=LOWPASS_FILTER_ORDER)
              
    ssf_transformed_vector = windowed_slope_sum_function(lp_filtered_vector, SSF_WINDOW_LENGTH)
    
    peak_thresholder = PeakThreshold(ssf_transformed_vector, sampling_rate)
    p_threshold = peak_thresholder.get_threshold()
    rolling_point_buffer = collections.deque(np.zeros(ROLLING_POINT_SPACING), maxlen=ROLLING_POINT_SPACING)  
    r_period_count = 0
    rising_edge = False
    
    # peak detection state machine
    for sample_num, bp_val in enumerate(ssf_transformed_vector):
        lrp = rolling_point_buffer[0] # left rolling point
        rrp = bp_val # right rolling point
                
        if rising_edge and lrp > rrp and lrp > p_threshold:
            r_period_count = 0
            peak_thresholder.update(lrp)
            p_threshold = peak_thresholder.get_threshold()
            R_wave_locations.append(sample_num - R_WAVE_DELAY_IN_SAMPLES - ROLLING_POINT_SPACING)
            rising_edge = False
        elif not rising_edge and r_period_count > REFRACTORY_PERIOD and lrp > rrp:
            rising_edge = True
        else:
            pass
            
        r_period_count += 1
        rolling_point_buffer.append(bp_val)
        
    if output_type=='sample_idx':
        return R_wave_locations
    elif output_type=='time':
        return np.array(R_wave_locations) / sampling_rate
    else:
        error_msg = "output_type: '{}' is invalid".format(output_type)
        raise ValueError(error_msg)

# TUNING / TEST METHODS
def filter_frequency_response_tester():
    """convenience method from http://stackoverflow.com/a/12233959/2754647
    plots the frequency responses of a scipy filter    
    """
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 250
    highcut = 16

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_lowpass(highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

def test():
    signal = []    
    with open('yorkshire-pig-trial1.log') as f:
        f.readline() # skip header
        for line in f:
            if line:
                signal_voltage = line.split(' ')[1]
                signal.append(float(signal_voltage))
    estimate_all_R_wave_locations(signal, SAMPLING_RATE)

if __name__ == '__main__':
    test()