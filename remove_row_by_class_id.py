import os

# Path to your dataset
dataset_dir = 'datasets/dumbbell'

# Iterate through all dataset splits
for split in ['train', 'valid', 'test']:
    labels_dir = os.path.join(dataset_dir, split, 'labels')
    
    # Skip if the labels directory doesn't exist
    if not os.path.isdir(labels_dir):
        continue

    # Loop through all label files
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_dir, filename)
            
            # Read the contents of the file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Keep only lines that start with class_id 0
            filtered_lines = [line for line in lines if line.strip().startswith('0 ')]

            # Overwrite the file with filtered lines
            with open(file_path, 'w') as f:
                f.writelines(filtered_lines)

print("All lines with class_id = 1 have been removed.")
