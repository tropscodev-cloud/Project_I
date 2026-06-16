import cv2
import os
import shutil

def shorten_video(input_path, output_path, max_frames=300):
    print(f"Shortening {input_path} to {output_path} (max {max_frames} frames)...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1
        
    cap.release()
    out.release()
    print(f"Finished. Wrote {count} frames.")
    return True

def main():
    data_dir = "data"
    cams = ["cam1", "cam2", "cam3"]
    
    for cam in cams:
        in_file = os.path.join(data_dir, f"{cam}.mp4")
        temp_file = os.path.join(data_dir, f"{cam}_temp.mp4")
        if os.path.exists(in_file):
            success = shorten_video(in_file, temp_file, max_frames=300)
            if success:
                # Replace original with the shortened version
                os.remove(in_file)
                os.rename(temp_file, in_file)
                print(f"Replaced {in_file} with shortened version.")
        else:
            print(f"File {in_file} not found, skipping.")
            
    # Delete unused cam4 to cam7
    unused_cams = ["cam4", "cam5", "cam6", "cam7"]
    for cam in unused_cams:
        unused_file = os.path.join(data_dir, f"{cam}.mp4")
        if os.path.exists(unused_file):
            print(f"Removing unused file: {unused_file}")
            os.remove(unused_file)

if __name__ == "__main__":
    main()
