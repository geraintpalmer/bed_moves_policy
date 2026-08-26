import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python count_bin.py <path_to_bin_file>")
    sys.exit(1)

filepath = sys.argv[1]
data = np.fromfile(filepath, dtype=np.int64)
print(len(data))