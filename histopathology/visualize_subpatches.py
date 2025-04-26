import numpy as np
import matplotlib.pyplot as plt
import os

patches_dir = os.path.join('output', 'patches')

# Get all patch files
patch_files = [f for f in os.listdir(patches_dir) if f.endswith('.npz')]
print(f"Number of patch files: {len(patch_files)}")

# Choose a random patch file
patch_file = np.random.choice(patch_files)
print(f"Random patch file: {patch_file}")
patch_path = os.path.join(patches_dir, patch_file)

# Load the patch
data = np.load(patch_path)
print(data)
mask = data['patch']

print(f"Patch shape: {mask.shape}")
print(f"Unique values in patch: {np.unique(mask)}")

# Display the patch
plt.figure(figsize=(8, 8))
plt.imshow(mask, cmap='gray', interpolation='nearest')
plt.title(f"Patch: {patch_file}")
plt.axis('off')
plt.show()
