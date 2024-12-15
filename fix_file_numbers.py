import os
import re
from pathlib import Path

# Get all files in stages directory that start with numbers
stages_dir = Path('./stages')
files = []
pattern = re.compile(r'^\d{2}_.*\.py$')

for f in stages_dir.iterdir():
    if pattern.match(f.name):
        files.append(f)

# Sort files alphabetically 
files.sort()

# Rename files with new numbers
for i, file in enumerate(files, start=1):
    new_name = f"{i:02d}{file.name[2:]}"
    file.rename(file.parent / new_name)
    print(f"Renamed {file.name} to {new_name}")
