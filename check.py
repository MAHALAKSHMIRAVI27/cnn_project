import os

path = "dataset"

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)

    print(folder, "=", len(os.listdir(folder_path)), "images")