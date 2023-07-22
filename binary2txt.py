import os
import cv2
import multiprocessing

input_dir = '/media/di0n/e065dccb-6e16-4544-b534-b315cae238b9/fiftyone/open-images-v7/train/labels/masks' # Your input folder containing all subfolders
output_dir = '/media/di0n/e065dccb-6e16-4544-b534-b315cae238b9/fiftyone/open-images-v7/train/labels/output' # The folder where you want the contents to be written

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

def process_file(folder, file):
    image_path = os.path.join(input_dir, folder, file)

    # Load the binary mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # Print the polygons
    with open('{}.txt'.format(os.path.join(output_dir, file)[:-4]), 'w') as f:
        for polygon in polygons:
            for p_, p in enumerate(polygon):
                if p_ == len(polygon) - 1:
                    f.write('{}\n'.format(p))
                elif p_ == 0:
                    f.write('0 {} '.format(p))
                else:
                    f.write('{} '.format(p))

    print(f"Processed {file}")

# Function to process all files in a folder
def process_folder(folder):
    folder_path = os.path.join(input_dir, folder)
    if not os.path.isdir(folder_path):
        return

    for file in os.listdir(folder_path):
        process_file(folder, file)

# Get all sub-folders in the input directory
sub_folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]
if len(sub_folders) == 0:
    sub_folders.append(input_dir)

# Create and start a process for each sub-folder
processes = []
for folder in sub_folders:
    process = multiprocessing.Process(target=process_folder, args=(folder,))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()

print("All files processed.")
