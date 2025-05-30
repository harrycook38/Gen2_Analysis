Analysis pipeline for our 2nd Generation NMOR gradiometer

Our sensor outputs semi-colon delimited CSV files for every channel (for example, phase, X, Y, PID shift)
When memory is reached, the CSV are saved in discrete files that must be concatenated

This pipeline will concatenate CSV if needed, it is then converted into a fif file such that it is compatible with MNE.

Preprocessing
  1. Preprocessing/concat.py (if needed)
  2. Preprocessing/generate_fif.py
    a. Select target stings such as 'shift', 'trigin1', 'auxin0'
          *some edits required for anything beyond shift and trigin
  3. Inspect time-series data from the fif file, and generate filtered version if desired
