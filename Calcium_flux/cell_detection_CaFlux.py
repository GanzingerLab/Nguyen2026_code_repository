# -*- coding: utf-8 -*-
import nd2
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import io, filters, measure, morphology, segmentation, color
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_holes,
    remove_small_objects, 
    binary_dilation, 
    binary_erosion, 
    isotropic_closing
)
from scipy.optimize import curve_fit
import tifffile
import pandas as pd
from skimage.feature import blob_dog
import csv
from skimage.draw import disk
from matplotlib.patches import Circle
import trackpy as tp
from glob import glob
import re

def phansalkar_threshold(image_stack, radius=15, k=0.25, p=2.0, q=10.0):
    """
        Apply Phansalkar local thresholding per frame.
    
        Parameters
        ----------
        image_stack : np.ndarray
            2D or 3D array of images (frames, height, width).
        radius : int, optional
            Local window radius, by default 15.
        k : float, optional
            Phansalkar parameter k, by default 0.25.
        p : float, optional
            Phansalkar parameter p, by default 2.0.
        q : float, optional
            Phansalkar parameter q, by default 10.0.
    
        Returns
        -------
        np.ndarray
            Binary thresholded image or stack of same shape.
"""
    window_size = (radius * 2) + 1

    def threshold_single(image):
        image = image / np.max(image) if np.max(image) > 0 else image
        mean = cv2.blur(image, (window_size, window_size))
        mean_sq = cv2.blur(image**2, (window_size, window_size))
        std = np.sqrt(mean_sq - mean**2)
        threshold = mean * (1 + p * np.exp(-q * mean) + k * ((std / 0.5) - 1))
        return image > threshold

    if image_stack.ndim == 2:
        return threshold_single(image_stack)
    elif image_stack.ndim == 3:
        # Process each frame independently and stack results
        binary_stack = np.zeros_like(image_stack, dtype=bool)
        for i in range(image_stack.shape[0]):
            binary_stack[i] = threshold_single(image_stack[i])
        return binary_stack
    else:
        raise ValueError("Input must be 2D or 3D numpy array")
def remove_small_objects_per_frame(stack, min_size=100, connectivity=1):
    """
        Remove small objects from each frame of a binary stack.
    
        Parameters
        ----------
        stack : np.ndarray
            Binary image stack (frames, height, width).
        min_size : int, optional
            Minimum object size in pixels to keep, by default 100.
        connectivity : int, optional
            Connectivity for object removal, by default 1.
    
        Returns
        -------
        np.ndarray
            Cleaned binary stack.
"""
    cleaned_stack = np.zeros_like(stack, dtype=bool)
    for i in range(stack.shape[0]):  # assuming frames on axis 0
        cleaned_stack[i] = remove_small_objects(stack[i], min_size=min_size, connectivity=connectivity)
    return cleaned_stack
def open_BF_image(file_path, save_stack = False):    
    def poly_func(coords, a, b, c, d, e, f):
        x, y = coords
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
    
    nd2_file = nd2.ND2File(file_path)
    
    # Access image data
    image_data = nd2_file.asarray()
    
   
    #2. Extract brigthfield channel
    ch1 = image_data[:, 1, :, :].astype(float)  # ensure float64
    fluo = image_data[:, 0, :, :].astype(float)  # ensure float64
    
    #3. Build mean background
        #make mean of the whole image
    avg_frame = np.mean(ch1, axis=0)
        #invert the background so dark things (clusters and out of focus cells are brigth)
    background = 1 - (avg_frame / 4059.0)
        #make a massive blur to eliminate cells and only keep the gradient
    background = filters.gaussian(background, sigma=16)

    #4. Fit 2D polynomial (quadratic)
        #make image shape
    h, w = background.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.vstack((xx.ravel(), yy.ravel()))
    
        # Downsample for speed
    sample_rate = 4
        #fit image to the plynomial function defined above --> we fit it so intensity values are mantained
    popt, _ = curve_fit(
        poly_func,
        coords[:, ::sample_rate],
        background.ravel()[::sample_rate]
    )
        #reshape fit into an image
    fitted_gradient = poly_func((xx, yy), *popt).reshape(h, w)
        #make again a gaussian blur to eliminate lines in different region of the gradient. 
    fitted_gradient = filters.gaussian(fitted_gradient, sigma=6)
    
    # --- 5. Correct frame (keep as float) ---
        #invert whole video for the same reasons as above
    frame_float = 1 - (ch1/ 4059.0)  # range ~0–1
        #remove background
    corrected = frame_float - fitted_gradient
    
    # Normalize within current range
        # make image use the whole pixel value space
    corrected -= corrected.min()
    corrected /= corrected.max()
    stack = (corrected * 255).astype(np.uint8)
    
        
    inv_stack = (255-stack)/255
    if save_stack: 
        # Extract the directory path
        directory_path = os.path.dirname(file_path)
        
        # Save the TIFF file in the same directory
        tifffile.imwrite(os.path.join(directory_path, "BF_no_background.tif"), stack, imagej=True, metadata={
            'axes': 'TYX',   # T = frames, Y = height, X = width
            'unit': 'pixel'
        })

    return inv_stack, fluo

def find_skip_regions(inv_stack): 
    blurred_stack = np.array([filters.gaussian(frame, sigma=2) for frame in inv_stack])
    threshold = filters.threshold_triangle(blurred_stack)
    binary = blurred_stack > threshold
    binary = filters.gaussian(binary, sigma=4)>0.1
    return binary_dilation(binary)


def filter_tracks(df, binary, scale_factor = 0.618050277012301):
    filtered_tracks = []
    for frame_idx in range(len(binary)):
        frame_tracks = df[df['frame'] == frame_idx]
        binary_mask = binary[frame_idx]
        
        filtered_positions = frame_tracks[
            ~binary_mask[
                (frame_tracks['y'] / scale_factor).astype(int),
                (frame_tracks['x'] / scale_factor).astype(int)
            ]
        ]
        
        filtered_tracks.append(filtered_positions)
    
    return pd.concat(filtered_tracks)


def filter_tracks2(file, binary, scale_factor = 0.618050277012301):
    df = pd.read_csv(file, index_col=False, low_memory=False)

    filtered_tracks = []
    for frame_idx in range(len(binary)):
        frame_tracks = df[df['FRAME'] == frame_idx]
        binary_mask = binary[frame_idx]
        
        filtered_positions = frame_tracks[
            ~binary_mask[
                (frame_tracks['POSITION_Y'] / scale_factor).astype(int),
                (frame_tracks['POSITION_X'] / scale_factor).astype(int)
            ]
        ]
        
        filtered_tracks.append(filtered_positions)
    
    return pd.concat(filtered_tracks)

def find_cells(fluo, diameter = 12, below_range= 0.8, ab_range= 1.15, threshold = 3, frame_int = 3.0, px2um = 1.618, plot = False):
    D = diameter *px2um #from um to px
    sigma_target = D / 2.828   # Transform diameter into radious of a gaussian: D / (2*sqrt(2))
    min_sigma = sigma_target*below_range
    max_sigma = sigma_target*ab_range
    
    rows = []
    spot_counter = 0  # unique spot IDs
    for frame in range(0, 301):
        img = fluo[frame]
    
        blobs = blob_dog(img, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=1.05, threshold=threshold)
        if plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap='gray')
        for (y, x, sigma) in blobs:
            # fixed circular ROI for intensity (robust for elongated cells)
            r = float(np.sqrt(2) * sigma) 
            if plot: 
                ax.add_patch(Circle((x, y), r, fill=False, linewidth=1.2, edgecolor='r'))
            rr, cc = disk((y, x), r, shape=img.shape)
            vals = img[rr, cc]
            if vals.size == 0:
                continue
    
            rows.append({
                "spot_id": int(spot_counter),
                "frame": int(frame),
                "x": float(x),
                "y": float(y),
                "sigma": float(sigma),
                "radius_px": float(r),
                "mean_intensity": float(vals.mean()),
                "median_intensity": float(np.median(vals)),
                "min_intensity": float(vals.min()),
                "max_intensity": float(vals.max()),
                "sum_intensity": float(vals.sum()),
                "std_intensity": float(vals.std(ddof=0))
            })
            spot_counter += 1
        if plot:
            ax.set_title(f"Frame {frame} — blobs: {len(blobs)}")
            ax.axis('off')
            plt.show()
            plt.close(fig)
    # Build detections DF (trackpy keeps extra columns, incl. spot_id)
    features = pd.DataFrame(rows)
    return features

def track_cells(features, search_range = 8, memory=2, px2um = 1.618, frame_int=3.0):
    tracks = tp.link_df(features, search_range=search_range,
                        memory=memory)#, predictor=predictor)
    
    # (Optional) filter very short tracks
    tracks = tp.filter_stubs(tracks, threshold=300*0.1).reset_index(drop=True)
    
    
    # Helper: safe multiply (pixels -> microns)
    def px_to_um(arr): return np.asarray(arr, dtype=float) * px2um
    
    # Build TrackMate-like DataFrame
    tm_like = pd.DataFrame({
        "LABEL":               tracks["spot_id"].apply(lambda i: f"ID{i}"),
        "ID":                  tracks["spot_id"],                               # Spot ID
        "TRACK_ID":            tracks.get("particle", pd.Series([-1]*len(tracks))),  # from trackpy
        "QUALITY":             np.nan,                                          # unknown here
        "POSITION_X":          px_to_um(tracks["x"]),                  # µm
        "POSITION_Y":          px_to_um(tracks["y"]),                  # µm
        "POSITION_Z":          np.zeros(len(tracks), dtype=float),              # set if 3D
        "POSITION_T":          tracks["frame"] * float(frame_int),       # seconds
        "FRAME":               tracks["frame"].astype(int),                     # frame index
        "RADIUS":              tracks["radius_px"] / px2um,             # µm
        "VISIBILITY":          1,                                               # 1 by default
        "MANUAL_SPOT_COLOR":   "",                                              # empty
        "MEAN_INTENSITY_CH1":  tracks["mean_intensity"],                        # counts
        "MEDIAN_INTENSITY_CH1":tracks["median_intensity"],                      # counts
        "MIN_INTENSITY_CH1":   tracks["min_intensity"],                         # counts
        "MAX_INTENSITY_CH1":   tracks["max_intensity"],                         # counts
        "TOTAL_INTENSITY_CH1": tracks["sum_intensity"],                         # counts
        "STD_INTENSITY_CH1":   tracks["std_intensity"],                         # counts
        "CONTRAST_CH1":        np.nan,                                          # not computed here
        "SNR_CH1":             np.nan                                           # not computed here
    })
    return tm_like

def save_tracks(tm_like, path, output_name = 'tracks_filtered.csv'):
    # (Optional) order the columns exactly as your sample
    tm_cols = ["LABEL","ID","TRACK_ID","QUALITY","POSITION_X","POSITION_Y","POSITION_Z",
               "POSITION_T","FRAME","RADIUS","VISIBILITY","MANUAL_SPOT_COLOR",
               "MEAN_INTENSITY_CH1","MEDIAN_INTENSITY_CH1","MIN_INTENSITY_CH1","MAX_INTENSITY_CH1",
               "TOTAL_INTENSITY_CH1","STD_INTENSITY_CH1","CONTRAST_CH1","SNR_CH1"]
    tm_like = tm_like[tm_cols]
    
    
    header_row1 = ["LABEL","ID","TRACK_ID","QUALITY","POSITION_X","POSITION_Y","POSITION_Z",
                   "POSITION_T","FRAME","RADIUS","VISIBILITY","MANUAL_SPOT_COLOR",
                   "MEAN_INTENSITY_CH1","MEDIAN_INTENSITY_CH1","MIN_INTENSITY_CH1",
                   "MAX_INTENSITY_CH1","TOTAL_INTENSITY_CH1","STD_INTENSITY_CH1",
                   "CONTRAST_CH1","SNR_CH1"]
    
    header_row2 = ["Label","Spot ID","Track ID","Quality","X","Y","Z","T","Frame","Radius",
                   "Visibility","Spot color","Mean intensity ch1","Median intensity ch1",
                   "Min intensity ch1","Max intensity ch1","Sum intensity ch1",
                   "Std intensity ch1","Contrast ch1","Signal/Noise ratio ch1"]
    
    header_row3 = ["","","","(quality)","(micron)","(micron)","(micron)","(sec)","",
                   "(micron)","","(counts)","(counts)","(counts)","(counts)",
                   "(counts)","(counts)","",""]
    
    # Save with custom header rows
    out_path = os.path.join(path, path, output_name)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(header_row1)
        writer.writerow(header_row2)
        writer.writerow(header_row3)
    tm_like.to_csv(out_path, sep=",", mode="a", index=False)
    print(f"Saved: {out_path}")
# %%


path = r'D:\Data\Chi_data\2. Ca flux'  # Root directory to scan
nd2_dirs = []

for root, dirs, files in os.walk(path):
    # Check if any .nd2 file exists in the current directory
    if any(file.lower().endswith('.nd2') for file in files):
        nd2_dirs.append(root)
del dirs, files, root

# %%Finding cells



px2um = 1.618    # px per um
frame_int = 3.0      # seconds per frame (set if you want POSITION_T)
expected_diameter = 12 #in um

search_range = 15
memory = 8

for i in nd2_dirs:
    print(i)
    file = os.path.join(i, 'tracks_unfiltered.csv')
    if os.path.isfile(file):
        print(f'skipping {file}') 
        continue
    else:
        file = glob(i + '/**.nd2', recursive=True)[0] 
        path = os.path.dirname(file)
        inv_stack, fluo = open_BF_image(file, save_stack=False)
        # binary = find_skip_regions(inv_stack[0:301])
        cells_filtered = find_cells(fluo[0:301], diameter = expected_diameter, px2um = px2um, frame_int=frame_int, plot = False)
        # cells_filtered = filter_tracks(cells, binary, 1)
        tracks = track_cells(cells_filtered, search_range = search_range, memory = memory)
        save_tracks(tracks, path, output_name = 'tracks_unfiltered.csv')
# %%
file = r"D:\Data\Chi_data\2. Ca flux\selfmade_chamber\Low expression CAR\3000xdilutedCD19\20250716\CART3Lo\R2\R2.nd2"

inv_stack, fluo = open_BF_image(file, save_stack=False)
binary = find_skip_regions(inv_stack)
plt.imshow(fluo[300])
plt.imshow(binary[300], alpha = 0.5)
