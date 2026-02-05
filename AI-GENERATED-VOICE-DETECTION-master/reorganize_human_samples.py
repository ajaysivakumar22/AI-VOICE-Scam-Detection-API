import os
import shutil
import re
import glob

SOURCE_DIR = r"C:\Users\ajays\Music\Samples_Hmannnnnnnnnnn_1"
TARGET_DIR = r"d:\GuviHCL_Hackathon\ai_voice_detection_api\training\data\human"

def get_highest_index(directory):
    max_idx = 0
    pattern = re.compile(r"HumanSample_(\d+)\.mpeg")
    if not os.path.exists(directory):
        return 0
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    if not os.path.exists(TARGET_DIR):
        try:
            os.makedirs(TARGET_DIR)
        except OSError as e:
            print(f"Error creating target directory {TARGET_DIR}: {e}")
            return

    source_files = glob.glob(os.path.join(SOURCE_DIR, "*.mpeg"))
    if not source_files:
        print(f"No .mpeg files found in {SOURCE_DIR}")
        return

    current_idx = get_highest_index(TARGET_DIR)
    print(f"Highest existing index: {current_idx}")

    count = 0
    for src_file in source_files:
        current_idx += 1
        # preserve at least 2 digits padding like HumanSample_01.mpeg
        new_filename = f"HumanSample_{current_idx:02d}.mpeg"
        
        dst_path = os.path.join(TARGET_DIR, new_filename)
        try:
            shutil.copy2(src_file, dst_path)
            print(f"Copied {os.path.basename(src_file)} -> {new_filename}")
            count += 1
        except Exception as e:
            print(f"Failed to copy {src_file}: {e}")
    
    print(f"Successfully copied and renamed {count} files.")

if __name__ == "__main__":
    main()
