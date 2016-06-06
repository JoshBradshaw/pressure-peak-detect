from __future__ import division
import collections
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt
import warnings

"""
Algorithm adapted from:
An Open-source Algorithm to Detect Onset of Arterial Blood Pressure Pulses 
by: W Zong, T Heldt, GB Moody, RG Mark 
"""
### USER TUNABLE FILTER SETTINGS
# Hz !!! MUST BE SET CORRECTLY FOR TIME OUTPUT TO BE ACCURATE
# otherwise setting it approximately will be sufficient for sample rates >50Hz
SAMPLING_RATE = 561
MAX_EXPECTED_HR = 320 # BPM -- used to set the refractory period

LOWPASS_FILTER_CUTTOFF_FREQUENCY = 16 # Hz
# effective filtering will be 2x this order, because the filter is applied twice
# see http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.signal.filtfilt.html
LOWPASS_FILTER_ORDER = 1  

### USER TUNABLE WAVEFORM TRANSFORM SETTINGS
# this value is used to determine the SSF window length. This paramater is not
# very sensitive and does not typically require adjustment
PULSE_RISE_TIME = 0.06 # seconds
# 128 ms window recommended by Zong et. Al for Adult BP. It is sometimes 
# necessary to shorten this window length when dealing with very fast heartrates
SSF_WINDOW_LENGTH = int(math.floor(SAMPLING_RATE * PULSE_RISE_TIME))

### USER TUNABLE PEAK THRESHOLDING SETTINGS
# BPM -- used to calculate the refractory period
# you must use a much higher value is the subject has arrythmia or skip-beats
REFRACTORY_PERIOD = int(math.floor((60 / MAX_EXPECTED_HR) * SAMPLING_RATE))

# the peak threshold is initialized as ~2-4 times the mean SSF signal value,
# then dyanamically updated as peaks are detected
INITIAL_PEAK_THRESHOLD = 2 # arbitrary coefficient -- should be 1-5

# to avoid false positive detections of peaks, only detect a peaks
# that exceed this fraction of the magnitude of the last 5 detected peaks 
PEAK_THRESHOLD_FRACTION = 0.6 # 0-1

# number of peaks to average the peak threshold value over. A higher number
# will decrease the risk of false positive detections, but increase the chance
# of false negative (misses) detections in the event of a sudden baseline change
PEAK_BUFFER_LEN = 3 # number of peaks

# the stability of the rolling point peak detection algorithm is improved
# marginally if a larger rolling point spacing is used.
ROLLING_POINT_SPACING = 4 # samples

# seek up to 1 second back to find the start of the pressure peak
MAXIMUM_TROUGH_SEEK_SAMPLES = SAMPLING_RATE

DEBUG = True # plot the algorithm output

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
    # using filtfilt makes this a zero phase filter
    y = filtfilt(b, a, data)
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
    # init with a monotonically increasing function    
    window_ring_buffer = collections.deque(np.zeros(window_length), maxlen=window_length)

    ssf_vector = []
    window_ssf = 0
    for ii, bp_sample in enumerate(fitered_bp_vector):
        window_ssf -= ssf(window_ring_buffer[0], window_ring_buffer[1])
        window_ssf += ssf(window_ring_buffer[-1], bp_sample)
        window_ring_buffer.append(bp_sample)
        if ii > window_length:
            ssf_vector.append(window_ssf)
        else:
            ssf_vector.append(0)
    return ssf_vector
    
class PeakThreshold(object):    
    """
    the peak threshold value is simply a rolling average of the the preceding
    peak values.      
    """    
    def __init__(self, ssf_vector, sampling_rate):
        # takes the mean of the SSF waveform
        # assumes that the SSF waveform is very bottom heavy ie. mean << max
        initial_threshold = INITIAL_PEAK_THRESHOLD * np.mean(ssf_vector)
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

def estimate_all_pressure_peak_locations(bp_vector, sampling_rate, backtrack_to_pulse_onset=False, output_type='sample_idx'):    
    """returns a list of the R wave locations, in samples
    
    output_types:
        sample_idx -- a list of the bp_vector sample indexes which correspond to R-wave locations
        time -- the times of the R wave locations in seconds, relative to the start of bp_vector
    """
    detected_locations = []  
    lp_filtered_vector = butter_lowpass_filter(bp_vector, LOWPASS_FILTER_CUTTOFF_FREQUENCY, 
                              sampling_rate, order=LOWPASS_FILTER_ORDER)
              
    ssf_transformed_vector = windowed_slope_sum_function(lp_filtered_vector, SSF_WINDOW_LENGTH)
    peak_thresholder = PeakThreshold(ssf_transformed_vector, sampling_rate)
    p_threshold = peak_thresholder.get_threshold()
    rolling_point_buffer = collections.deque(np.zeros(ROLLING_POINT_SPACING), maxlen=ROLLING_POINT_SPACING)  
    r_period_count = REFRACTORY_PERIOD
    rising_edge = False
    
    # peak detection state machine
    for sample_num, bp_val in enumerate(ssf_transformed_vector):
        lrp = rolling_point_buffer[0] # left rolling point
        rrp = bp_val # right rolling point
                
        if rising_edge and lrp > rrp:
            r_period_count = 0
            peak_thresholder.update(lrp)
            p_threshold = peak_thresholder.get_threshold()
            rising_edge = False
            
            if backtrack_to_pulse_onset:
                detection_sample_num = find_onset_of_trough(lp_filtered_vector, sample_num, ssf_transformed_vector, p_threshold, sampling_rate)
            else:
                detection_sample_num = sample_num - ROLLING_POINT_SPACING
            # detection_sample_num is None if find_onset_of_trough backtracking fails
            if detection_sample_num is not None:
                detected_locations.append(detection_sample_num)
        elif not rising_edge and r_period_count > REFRACTORY_PERIOD and rrp > lrp and lrp > p_threshold:
            rising_edge = True
        else:
            pass # state unchanged during this step
        
        r_period_count += 1
        rolling_point_buffer.append(bp_val)
    
    # edge case where the recording ends on a rising edge, but cuts off before the peak
    if backtrack_to_pulse_onset and rising_edge:
        find_onset_of_trough(lp_filtered_vector, sample_num, ssf_transformed_vector, p_threshold, sampling_rate)
        
    if DEBUG:
        # display the intermediate filtered and SSF waveforms, for testing
        x = np.linspace(0, len(bp_vector), len(bp_vector))
        plt.figure(1)
        plt.plot(x, bp_vector)
        x = np.arange(0, len(bp_vector))
        plt.plot(x, lp_filtered_vector)
        plt.plot(x[:len(ssf_transformed_vector)], ssf_transformed_vector)
        
        bp_min = min(bp_vector)
        bp_max = max(bp_vector)
        detections_x = []
        detections_y = []
        for sample_num in detected_locations:
            detections_x.append(sample_num)
            detections_y.append(bp_vector[sample_num])
        plt.title("BP, FILTERED BP, SSF TRANSFORMED BP, AND DETECTIONS")
        plt.ylabel("Voltage, or mmHg, or whatever BP was measured in")
        plt.xlabel("Samples")
        plt.plot(detections_x, detections_y, 'ro')
    
    if output_type=='sample_idx':
        return detected_locations
    elif output_type=='time':
        return np.array(detected_locations) / sampling_rate
    else:
        error_msg = "output_type: '{}' is invalid".format(output_type)
        raise ValueError(error_msg)
        
def find_onset_of_trough(lp_filtered_vector, pressure_peak_sample_num, ssf_vector, ssf_threshold, sampling_rate):
    """
    given the location of a pressure peak, searches backwards along the rising
    edge to find the rightmost edge of the trough between peaks    
    """
    trough_onset_locations = []            

    backwards_seek_range = reversed(xrange(pressure_peak_sample_num-MAXIMUM_TROUGH_SEEK_SAMPLES, pressure_peak_sample_num))
    rolling_point_buffer = collections.deque(np.zeros(ROLLING_POINT_SPACING), maxlen=ROLLING_POINT_SPACING)    
    rising_edge = False
    
    for ii in backwards_seek_range:
        if ii < 0:
            msg = "Unable to locate the beginning of the first pressure peak, algorithm reached start of recording"
            warnings.warn(msg)
            return None
        
        rrp = rolling_point_buffer[0] # left rolling point
        lrp = lp_filtered_vector[ii]  # right rolling point
        rolling_point_buffer.append(lrp)
        
        if rising_edge and lrp > rrp:
            pressure_peak_location = ii + ROLLING_POINT_SPACING
            trough_onset_locations.append(ii)
            return pressure_peak_location
        elif not rising_edge and lrp < rrp and ssf_vector[ii] < ssf_threshold:
            rising_edge = True
        else:
            pass
        
    msg = "Unable to find the start of the pressure peak at time: '{}'".format(pressure_peak_sample_num / sampling_rate)
    warnings.warn(msg)
    return None

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
    plt.figure(2)
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
    skip_num = 1 #sometimes there is an abrupt jump at the first sample
    us_trace = np.load("trace_IMG_20160510_1_12_anon.dcm.npy")
    signal = us_trace[0][1][skip_num:]
    signal_time = us_trace[0][0][skip_num:]
	
    print "sampling rate is 1 sample per %f seconds = %f samples per second" % (signal_time[1] - signal_time[0], 1/(signal_time[1] - signal_time[0]))
	
    if False:
        plt.figure(1)
        plt.clf()
        plt.plot(signal_time, signal)

    L = estimate_all_pressure_peak_locations(signal, SAMPLING_RATE, True, 'time')
    print "R-wave time points = %s" % L
    
    plt.show()

if __name__ == '__main__':
    test()