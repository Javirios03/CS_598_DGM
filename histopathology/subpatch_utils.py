import os
import numpy as np

patches_path = os.path.join('output', 'patches')

# Print number of files
print(f"Number of files in {patches_path}: {len(os.listdir(patches_path))}")

# Files are NPZ with the structure patch_xstart_ystart_rowX_colY.npz, with row and col between 0 and 15
xstarts = {}
ystarts = {}
for filename in os.listdir(patches_path):
    if filename.endswith('.npz'):
        parts = filename.split('_')
        xstart = int(parts[1])
        ystart = int(parts[2])
        if xstart not in xstarts:
            xstarts[xstart] = 1
        else:
            xstarts[xstart] += 1
        if ystart not in ystarts:
            ystarts[ystart] = 1
        else:
            ystarts[ystart] += 1

print(f"Number of unique xstarts: {len(xstarts)}")
print(f"Number of unique ystarts: {len(ystarts)}")

# Check if all xstarts and ystarts have the same number of patches. Map xstart to number of patches
diff_xstarts = dict()
diff_ystarts = dict()

for xstart, count in xstarts.items():
    if count not in diff_xstarts:
        diff_xstarts[count] = [xstart]
    else:
        diff_xstarts[count].append(xstart)
for ystart, count in ystarts.items():
    if count not in diff_ystarts:
        diff_ystarts[count] = [ystart]
    else:
        diff_ystarts[count].append(ystart)

print(f"Number of unique xstart counts: {len(diff_xstarts)}")
print(f"Number of unique ystart counts: {len(diff_ystarts)}")
print(f"Unique xstart counts: {diff_xstarts}")
print(f"Unique ystart counts: {diff_ystarts}")

# Get empty patches
empty_patches = []
for filename in os.listdir(patches_path):
    if filename.endswith('.npz'):
        loaded_patch = np.load(os.path.join(patches_path, filename))
        patch = loaded_patch['patch']
        if np.count_nonzero(patch) == 0:
            empty_patches.append(filename)

print(f"Number of empty patches: {len(empty_patches)}")
print(f"This represents {len(empty_patches) / len(os.listdir(patches_path)) * 100:.2f}% of the total patches.")
