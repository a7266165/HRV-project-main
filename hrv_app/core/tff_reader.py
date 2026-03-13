"""
Module for reading ME6000 .tff format files.

http://www.biomation.com/kin/me6000.htm

"""
import datetime
import os
import struct

import numpy as np


def read_tff_header(file_path):
    """
    Read only the header of a TFF file (fast, no signal data).

    Returns
    -------
    result : dict
        Dictionary with keys: 'fs', 'n_sig', 'sig_name', 'base_time', 'base_date'
    """
    with open(file_path, 'rb') as fp:
        fields, _ = _rdheader(fp)
    return {
        'fs': fields['fs'],
        'n_sig': fields['n_sig'],
        'sig_name': fields['sig_name'],
        'base_time': fields['base_time'],
        'base_date': fields['base_date'],
    }


def read_tff_file(file_path):
    """
    High-level wrapper for reading a TFF file.

    Parameters
    ----------
    file_path : str
        Path to the .tff file.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'signal': ndarray, shape (n_samples, n_channels)
        - 'fs': int, sampling frequency
        - 'n_sig': int, number of signals
        - 'sig_name': list of str, channel names
        - 'base_time': datetime.time
        - 'base_date': datetime.date
        - 'markers': ndarray
        - 'triggers': ndarray
    """
    signal, fields, markers, triggers = rdtff(file_path)
    return {
        'signal': signal,
        'fs': fields['fs'],
        'n_sig': fields['n_sig'],
        'sig_name': fields['sig_name'],
        'base_time': fields['base_time'],
        'base_date': fields['base_date'],
        'markers': markers,
        'triggers': triggers,
    }


def rdtff(file_name, cut_end=False):
    """
    Read values from a tff file.

    Parameters
    ----------
    file_name : str
        Name of the .tff file to read.
    cut_end : bool, optional
        If True, cuts out the last sample for all channels. This is for
        reading files which appear to terminate with the incorrect
        number of samples (ie. sample not present for all channels).

    Returns
    -------
    signal : ndarray
        A 2d numpy array storing the physical signals from the record.
    fields : dict
        A dictionary containing several key attributes of the read record.
    markers : ndarray
        A 1d numpy array storing the marker locations.
    triggers : ndarray
        A 1d numpy array storing the trigger locations.
    """
    file_size = os.path.getsize(file_name)
    with open(file_name, 'rb') as fp:
        fields, file_fields = _rdheader(fp)
        signal, markers, triggers = _rdsignal(fp, file_size=file_size,
                                              header_size=file_fields['header_size'],
                                              n_sig=file_fields['n_sig'],
                                              bit_width=file_fields['bit_width'],
                                              is_signed=file_fields['is_signed'],
                                              cut_end=cut_end)
    return signal, fields, markers, triggers


def _rdheader(fp):
    tag = None
    while tag != 2:
        tag = struct.unpack('>H', fp.read(2))[0]
        data_size = struct.unpack('>H', fp.read(2))[0]
        pad_len = (4 - (data_size % 4)) % 4
        pos = fp.tell()
        if tag == 1001:
            storage_method = fs = struct.unpack('B', fp.read(1))[0]
            storage_method = {0:'recording', 1:'manual', 2:'online'}[storage_method]
        elif tag == 1003:
            fs = struct.unpack('>H', fp.read(2))[0]
        elif tag == 1007:
            n_sig = data_size
            channel_data = struct.unpack('>%dB' % data_size, fp.read(data_size))
            channel_map = ((1, 1, 'emg'),
                           (15, 30, 'goniometer'), (31, 46, 'accelerometer'),
                           (47, 62, 'inclinometer'),
                           (63, 78, 'polar_interface'), (79, 94, 'ecg'),
                           (95, 110, 'torque'), (111, 126, 'gyrometer'),
                           (127, 142, 'sensor'))
            sig_name = []
            for data in channel_data:
                base_name = 'unknown'
                if data == 0:
                    n_sig -= 1
                    break
                for item in channel_map:
                    if item[0] <= data <= item[1]:
                        base_name = item[2]
                        break
                existing_count = [base_name in name for name in sig_name].count(True)
                sig_name.append('%s_%d' % (base_name, existing_count))
        elif tag == 1009:
            display_scale = struct.unpack('>I', fp.read(4))[0]
        elif tag == 3:
            sample_fmt = struct.unpack('B', fp.read(1))[0]
            is_signed = bool(sample_fmt >> 7)
            bit_width = sample_fmt & 127
        elif tag == 101:
            n_seconds = struct.unpack('>I', fp.read(4))[0]
            base_datetime = datetime.datetime.utcfromtimestamp(n_seconds)
            base_date = base_datetime.date()
            base_time = base_datetime.time()
        elif tag == 102:
            n_minutes = struct.unpack('>h', fp.read(2))[0]
        fp.seek(pos + data_size + pad_len)
    header_size = fp.tell()
    fields = {'fs':fs, 'n_sig':n_sig, 'sig_name':sig_name,
              'base_time':base_time, 'base_date':base_date}
    file_fields = {'header_size':header_size, 'n_sig':n_sig,
                   'bit_width':bit_width, 'is_signed':is_signed}
    return fields, file_fields


def _rdsignal(fp, file_size, header_size, n_sig, bit_width, is_signed, cut_end):
    fp.seek(header_size)
    signal_size = file_size - header_size
    byte_width = int(bit_width / 8)
    dtype = str(byte_width)
    if is_signed:
        dtype = 'i' + dtype
    else:
        dtype = 'u' + dtype
    dtype = '>' + dtype
    max_samples = int(signal_size / byte_width)
    max_samples = max_samples - max_samples % n_sig
    signal = np.empty(max_samples, dtype=dtype)
    markers = []
    triggers = []
    sample_num = 0

    if cut_end:
        stop_byte = file_size - n_sig * byte_width + 1
        while fp.tell() < stop_byte:
            chunk = fp.read(2)
            sample_num = _get_sample(fp, chunk, n_sig, dtype, signal, markers, triggers, sample_num)
    else:
        while True:
            chunk = fp.read(2)
            if not chunk:
                break
            sample_num = _get_sample(fp, chunk, n_sig, dtype, signal, markers, triggers, sample_num)

    signal = signal[:sample_num]
    signal = signal.reshape((-1, n_sig))
    markers = np.array(markers, dtype='int')
    triggers = np.array(triggers, dtype='int')
    return signal, markers, triggers


def _get_sample(fp, chunk, n_sig, dtype, signal, markers, triggers, sample_num):
    tag = struct.unpack('>h', chunk)[0]
    if tag == -32768:
        escape_type = struct.unpack('B', fp.read(1))[0]
        data_len = struct.unpack('B', fp.read(1))[0]
        if escape_type == 1:
            markers.append(sample_num / n_sig)
        elif escape_type == 2:
            triggers.append(sample_num / n_sig)
        fp.seek(data_len + data_len % 2, 1)
    else:
        fp.seek(-2, 1)
        signal[sample_num:sample_num + n_sig] = np.fromfile(
            fp, dtype=dtype, count=n_sig)
        sample_num += n_sig
    return sample_num
