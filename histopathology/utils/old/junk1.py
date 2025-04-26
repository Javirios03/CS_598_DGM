import os

path = 'TCGA-XF-AAN2-01Z-00-DX1.EB523A3A-0DE0-4FFC-9FE7-CF4FB2FB36CF.svs'

# Export to a txt file the name of all files in the directory, sorting them by name
with open('output.txt', 'w') as f:
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            f.write(file + '\n')
            