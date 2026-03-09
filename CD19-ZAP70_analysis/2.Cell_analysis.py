#%%
import os
from postSPIT import tirf_analysis as ta
import pandas as pd 
from tensorflow.keras.models import load_model

#%%
#load the object to run the analysis for all FOV within a folder. Ch0 is CD19, Ch1 is ZAP70.  
data = ta.Dataset_combined_analysis(r'D:\Data\Chi_data\20250801_filtered')
#Runs combine_spots_clusters. This will:
# - run cluster detection anad analysis if it has not been done. 
# - Excludes spots detected by SPIT within the clusters. 
# - Calculates intensity metrics for the spots. 
data.combine_spots_clusters(ch = 'ch0',  save_videos=True)
data.combine_spots_clusters(ch = 'ch1', q = 8,  save_videos=True)
# tracks again the data with the new filtered spots. 
data.retrack()
# colocalizes tracks again with the new filtereds spots/tracks. 
data.recoloc_tracks()
# Load and use pre-trained model to analyze the maturation state of the immune synapses per frame. 
model = load_model(r'C:\Users\castrolinares\Data analysis\SPIT_G\ML_mature\V2\DL\MobileNetV2_expandedDataSet_V8.h5')
data.predict_maturation(model, save_plot = True, ch = 'ch1')
#%%
#Generate file with colocalization tracks longer than 10 frames
run_paths = data.run_paths
results = []
for i in run_paths:
    try:
        i_output = i.replace('_filtered', r'_filtered\output')
        cotracks_path = os.path.join(i_output, 'cluster_analysis_spots_filtered', '638nm_roi_locs_nm_trackpy_ColocsTracks_stats.hdf')
        cotracks = pd.read_hdf(cotracks_path)
        cotracks['folder'] = i_output
        results.append(cotracks.loc[cotracks.num_frames_coloc >=10, ['folder', 'colocID']])
    except:
        continue
final_results  = pd.concat(results)
final_results.to_csv(r'D:\Data\Chi_data\20250801_filtered\output\cotracks_longer_10frames.csv')