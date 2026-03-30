"""
Module for reading ME6000 .tff format files.

http://www.biomation.com/kin/me6000.htm

"""
import datetime
import os
import struct

import numpy as np


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
    file_size = os.path.getsize(file_name)
    with open(file_name, 'rb') as fp:
        fields, file_fields = _rdheader(fp)
    
        signal, markers, triggers = _rdsignal(
            fp, 
            file_size=file_size,
            header_size=file_fields['header_size'],
            n_sig=file_fields['n_sig'],
            bit_width=file_fields['bit_width'],
            is_signed=file_fields['is_signed'],
            cut_end=cut_end
        )
    
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
    
    dtype_str = str(byte_width)
    if is_signed:
        dtype_str = 'i' + dtype_str
    else:
        dtype_str = 'u' + dtype_str
    dtype_str = '>' + dtype_str
    
    # 1. 一次性讀取所有二進位資料到記憶體 (避免幾百萬次的磁碟 I/O)
    raw_bytes = fp.read(signal_size)
    
    # 2. 轉換為 NumPy 陣列以利快速檢查
    raw_data = np.frombuffer(raw_bytes, dtype=dtype_str)
    
    # 檢查是否存在 -32768 這個跳脫標記
    if -32768 not in raw_data:
        # -----------------------------------------------------------------
        # 【極速路徑】：檔案中沒有 markers，直接 reshape 返回 (瞬間完成)
        # -----------------------------------------------------------------
        n_samples = len(raw_data) - (len(raw_data) % n_sig)
        signal = raw_data[:n_samples].reshape((-1, n_sig)).copy()
        markers = np.array([], dtype='int')
        triggers = np.array([], dtype='int')
        
        if cut_end and signal.shape[0] > 0:
            signal = signal[:-1]
            
        return signal, markers, triggers
        
    else:
        # -----------------------------------------------------------------
        # 【備用路徑】：檔案中有 markers，使用記憶體內指標解析 (避開錯位問題)
        # -----------------------------------------------------------------
        max_samples = int(signal_size / byte_width)
        max_samples = max_samples - max_samples % n_sig
        
        # 直接建立正確形狀的 2D Array
        signal = np.empty((max_samples // n_sig, n_sig), dtype=dtype_str)
        markers = []
        triggers = []
        
        idx = 0
        sample_idx = 0  # 以完整樣本(包含所有通道)為單位的 index
        length = len(raw_bytes)
        
        # 預先編譯 struct 提升迴圈內速度
        unpack_h = struct.Struct('>h').unpack_from
        unpack_bb = struct.Struct('BB').unpack_from
        block_size = n_sig * byte_width
        
        while idx < length - 1:
            tag = unpack_h(raw_bytes, idx)[0]
            if tag == -32768:
                if idx + 4 > length:
                    break  # 標記資料不完整
                escape_type, data_len = unpack_bb(raw_bytes, idx + 2)
                if escape_type == 1:
                    markers.append(sample_idx)
                elif escape_type == 2:
                    triggers.append(sample_idx)
                
                # 跳過這個 marker 區塊: 2 (tag) + 2 (type & len) + data_len + padding
                skip = 4 + data_len + (data_len % 2)
                idx += skip
            else:
                if idx + block_size > length:
                    break  # 剩餘資料不足一個完整樣本
                
                # 直接從 buffer 拷貝一個完整的樣本區塊到 signal 矩陣
                signal[sample_idx, :] = np.frombuffer(raw_bytes, dtype=dtype_str, count=n_sig, offset=idx)
                sample_idx += 1
                idx += block_size
                
        signal = signal[:sample_idx]
        markers = np.array(markers, dtype='int')
        triggers = np.array(triggers, dtype='int')
        
        if cut_end and signal.shape[0] > 0:
            signal = signal[:-1]
            
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
