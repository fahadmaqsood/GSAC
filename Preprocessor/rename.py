import os

# Specify the folder containing the images
folder_path = "/home/rendong/3DGS/ExAvatar_RELEASE/fitting/data/Custom/data/female4_test/frames/"

# Specify the name of the output text file
output_txt_file = os.path.join("/home/rendong/3DGS/ExAvatar_RELEASE/fitting/data/Custom/data/female4_test/", "frame_list_all.txt")
output_txt_file1 = os.path.join("/home/rendong/3DGS/ExAvatar_RELEASE/fitting/data/Custom/data/female4_test/", "frame_list_train.txt")

# Get a sorted list of all files in the folder
files = sorted(f for f in os.listdir(folder_path) )

# Open the text file for writing
with open(output_txt_file, "w") as txt_file:
    for i, file_name in enumerate(files, start=0):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{i}.png"
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        
        # Write the new name (without extension) to the text file
        txt_file.write(f"{i}\n")
        
        print(f"Renamed {file_name} to {new_name}")

print(f"Renaming complete! List of new names saved to {output_txt_file}.")
