import os
import glob
import numpy as np
import tifffile
USE_TIFFFILE = True
import imageio
INPUT_DIR = '/data/Data/cropped_videos_for_beast'
OUTPUT_DIR = '/data/Data/cropped_videos_for_beast_mp4'

def get_tif_fps(tif_file):
    """
    Extract FPS from TIF file metadata.
    
    Args:
        tif_file: Path to input TIF file
        
    Returns:
        fps: Frames per second, or None if not found
    """
    fps = None
    
    # Try to get FPS from tifffile metadata
    try:
        with tifffile.TiffFile(tif_file) as tif:
            # Check ImageJ metadata
            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                # ImageJ stores fps as 'finterval' or 'fps'
                if 'finterval' in tif.imagej_metadata:
                    fps = 1.0 / tif.imagej_metadata['finterval']
                elif 'fps' in tif.imagej_metadata:
                    fps = tif.imagej_metadata['fps']
            
            # Check other metadata tags
            if fps is None and hasattr(tif, 'pages') and len(tif.pages) > 0:
                page = tif.pages[0]
                # Check for custom tags that might contain fps
                if hasattr(page, 'tags'):
                    for tag in page.tags.values():
                        if 'fps' in str(tag.name).lower() or 'frame' in str(tag.name).lower():
                            try:
                                fps = float(tag.value)
                                break
                            except (ValueError, TypeError):
                                pass
    except Exception as e:
        print(f"Warning: Could not read FPS from tifffile metadata: {e}")
    
    # Try imageio as fallback
    if fps is None:
        try:
            reader = imageio.get_reader(tif_file)
            meta = reader.get_meta_data()
            if 'fps' in meta:
                fps = meta['fps']
            reader.close()
        except Exception as e:
            print(f"Warning: Could not read FPS from imageio metadata: {e}")
    
    return fps

def convert_tif_to_mp4(tif_file, output_file, fps=None):
    """
    Convert a TIF file (image sequence) to MP4 video.
    
    Args:
        tif_file: Path to input TIF file
        output_file: Path to output MP4 file
        fps: Frames per second for output video (if None, will try to extract from TIF)
    """
    try:
        print(f"Converting {os.path.basename(tif_file)} to {os.path.basename(output_file)}...")
        
        # Get FPS from TIF file if not provided
        if fps is None:
            fps = get_tif_fps(tif_file)
            if fps is None:
                print("Warning: Could not extract FPS from TIF file. Using default 30 fps.")
                fps = 30
            else:
                print(f"Extracted FPS from TIF: {fps:.2f} fps")
        else:
            print(f"Using provided FPS: {fps:.2f} fps")
        # Read TIF file
        if USE_TIFFFILE:
            # tifffile is better for multi-page TIFs
            frames = tifffile.imread(tif_file)
            # Handle single frame case
            if frames.ndim == 2:
                frames = frames[np.newaxis, ...]
        else:
            # Fallback to imageio
            reader = imageio.get_reader(tif_file)
            frames = [frame for frame in reader]
            reader.close()
            frames = np.array(frames)
        
        # Create writer for MP4
        writer = imageio.get_writer(
            output_file, 
            fps=fps, 
            codec='libx264', 
            quality=8,
            pixelformat='yuv420p'  # Ensures compatibility
        )
        
        # Write frames
        for i, frame in enumerate(frames):
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                # Normalize to 0-255 range
                frame = frame.astype(np.float32)
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255
                frame = frame.astype(np.uint8)
            
            # Handle grayscale to RGB conversion if needed
            if len(frame.shape) == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                frame = np.repeat(frame, 3, axis=2)
            
            writer.append_data(frame)
        
        writer.close()
        
        print(f"Successfully converted {os.path.basename(tif_file)} ({len(frames)} frames)")
        
    except Exception as e:
        print(f"Error converting {tif_file}: {str(e)}")
        raise

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all tif files in the input directory
tif_files = glob.glob(os.path.join(INPUT_DIR, '*.tif'))

if not tif_files:
    print(f"No TIF files found in {INPUT_DIR}")
else:
    print(f"Found {len(tif_files)} TIF file(s) to convert")

# Convert each tif file to mp4
for tif_file in tif_files:
    # Get the base name of the tif file
    base_name = os.path.basename(tif_file)
    # Convert the tif file to mp4
    output_path = os.path.join(OUTPUT_DIR, base_name.replace('.tif', '.mp4'))
    
    convert_tif_to_mp4(tif_file, output_path)

print("Conversion complete!")
