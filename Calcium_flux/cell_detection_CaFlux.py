#%%
import nd2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import  filters
from scipy.optimize import curve_fit
import tifffile
import pandas as pd
from skimage.feature import blob_dog
import csv
from skimage.draw import disk
from matplotlib.patches import Circle
import trackpy as tp
from glob import glob

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

def find_cells(fluo, diameter = 12, below_range= 0.8, ab_range= 1.15, threshold = 3, frame_int = 3.0, px2um = 1.618, plot = False):
    D = diameter *px2um #from um to px
    sigma_target = D / 2.828   # Transform diameter into radius of a gaussian: D / (2*sqrt(2))
    min_sigma = sigma_target*below_range
    max_sigma = sigma_target*ab_range
    
    rows = []
    spot_counter = 0  # unique spot IDs
    for frame in range(0, 301):
        img = fluo[frame]
        # find blobs with difference of gaussian. 
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
    
    
    # pixels -> microns
    def px_to_um(arr): return np.asarray(arr, dtype=float) * px2um
    
    # Build TrackMate-like DataFrame - for compatibility with Ca2+ analysis from the lab
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
    # (Optional) order the columns 
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
# %%Find paths
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

search_range = 15 #for tracking
memory = 8 # for tracking

for i in nd2_dirs[0:1]:
    file = glob(i + '/**.nd2', recursive=True)[0] 
    path = os.path.dirname(file)
    #open images
    inv_stack, fluo = open_BF_image(file, save_stack=False)
    #find cells in the first 300 rames
    cells_filtered = find_cells(fluo[0:301], diameter = expected_diameter, px2um = px2um, frame_int=frame_int, plot = False)
    #tracks the cells
    tracks = track_cells(cells_filtered, search_range = search_range, memory = memory)
    #save the tracks for analysis in Analysis_script.py
    # save_tracks(tracks, path, output_name = 'tracks_unfiltered.csv')