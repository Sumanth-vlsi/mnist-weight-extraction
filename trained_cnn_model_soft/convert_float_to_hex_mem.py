# convert_multiple_float_files.py

import numpy as np
import os
import sys

def float_to_hex(val):
    return f"{np.float32(val).view(np.uint32):08x}"

def convert_file(input_file):
    output_file = os.path.splitext(input_file)[0] + ".mem"

    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    hex_lines = []
    for line in lines:
        try:
            num = float(line)
            hex_lines.append(float_to_hex(num))
        except ValueError:
            print(f"[!] Skipping invalid line: {line}")

    with open(output_file, "w") as f:
        f.write("\n".join(hex_lines))

    print(f"[✔] {input_file} → {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_multiple_float_files.py <file1.txt> <file2.txt> ...")
        sys.exit(1)

    for txt_file in sys.argv[1:]:
        convert_file(txt_file)
