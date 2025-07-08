import os
import csv
from collections import defaultdict

# Define your input/output paths
input_folder = r'W:\Data\2025_07_08-H_brain\braingrad1_000'
output_folder = r'W:\Data\2025_07_08-H_brain\braingrad1_000\concat'
os.makedirs(output_folder, exist_ok=True)

# Group files by base name (excluding the numeric suffix)
file_groups = defaultdict(list)
for fname in sorted(os.listdir(input_folder)):
    if fname.endswith('.csv'):
        # Extract base name (everything before _00000, _00001 etc.)
        base = fname.rsplit('_', 1)[0]
        file_groups[base].append(fname)

# Process each group
for base, files in file_groups.items():
    output_path = os.path.join(output_folder, f"{os.path.basename(base)}.csv")
    with open(output_path, 'w', newline='', encoding='utf-8') as fout:
        writer = None
        for i, fname in enumerate(sorted(files)):
            fpath = os.path.join(input_folder, fname)
            with open(fpath, 'r', encoding='utf-8') as fin:
                reader = csv.reader(fin, delimiter=';')
                if i == 0:
                    # Write header from the first file
                    header = next(reader)
                    writer = csv.writer(fout, delimiter=';')
                    writer.writerow(header)
                else:
                    next(reader)  # Skip header in subsequent files
                for row in reader:
                    writer.writerow(row)

print("Concatenation complete.")
