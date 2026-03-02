#%%

"""
Created on Fri Aug 22 09:54:03 2025

@author: castrolinares
"""

from postSPIT import plotting_classes as plc
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from natsort import natsorted
from tensorflow.keras.models import load_model

#%%
#load the object to run the analysis
data = plc.Dataset_combined_analysis(r'D:\Data\Chi_data\20250801_filtered', ch0_hint='638nm', ch1_hint='488nm')
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
data.predict_maturation(model, save_plot = True, ch = 'ch1', )
