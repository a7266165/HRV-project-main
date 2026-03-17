"""Quick script to plot TFF signal (partial read: 100~104 sec)."""
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np

from hrv_app.core.tff_reader import read_tff_header

FILE_PATH = r'C:\Users\莊淯任\Desktop\HRV-project-main\test_20260316.TFF'

# Read header to get fs and channel info
header = read_tff_header(FILE_PATH)
fs = header['fs']
n_sig = header['n_sig']

# Only read 100~104 sec range directly from binary
t_start, t_end = 100, 102
sample_start = int(t_start * fs)
sample_end = int(t_end * fs)

# Calculate byte offset: header is read by _rdheader, signal starts after
# Each sample group = n_sig * 2 bytes (16-bit signed big-endian)
import struct

# Re-read header to find header_size
with open(FILE_PATH, 'rb') as fp:
    tag = None
    while tag != 2:
        tag = struct.unpack('>H', fp.read(2))[0]
        data_size = struct.unpack('>H', fp.read(2))[0]
        pad_len = (4 - (data_size % 4)) % 4
        fp.seek(fp.tell() + data_size + pad_len)
    header_size = fp.tell()

    # Seek to the start sample position
    byte_offset = header_size + sample_start * n_sig * 2
    n_samples = (sample_end - sample_start)
    fp.seek(byte_offset)
    raw = np.frombuffer(fp.read(n_samples * n_sig * 2), dtype='>i2')

signal = raw.reshape((-1, n_sig))
time = np.linspace(t_start, t_end, signal.shape[0])

# Plot channel 0 and channel 1
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for ch in range(2):
    ax = axes[ch]
    ax.plot(time, signal[:, ch], linewidth=0.5)
    ax.set_ylabel(f'Channel {ch}')
    ax.set_title(header['sig_name'][ch])

axes[-1].set_xlabel('Time (sec)')
fig.suptitle(f'TFF Signal — fs={fs} Hz, showing {t_start}~{t_end}s', fontsize=13)
plt.tight_layout()
plt.show()
