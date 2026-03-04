#%%
# STEP 0: set the proper parameters in the auxiliary settings objects. 
from spit import settings 
from spit.SPIT import SPIT_Run, SPIT_Dataset, localize_tiff_run, localize_tiff_dataset

class RegistrationSettings:
    def __init__(self):
        self.registration_folder = r'C:\Users\castrolinares\Data analysis\SPIT_G\Raquel_6Feb2024\regis' #Folder containing the H-matrices and crop coordinates needed for the alignment of the channels
        self.verticalROI = [0, 1100] #Specify the heigth of the channels that you want to use. I am pretty sure that if we set it larger, it still does it correctly. 
        self.to_keep = [0, None] #number of frames to procees. 
        #[0, None] means all frames, if you want to change it, set the specific number ([0:200] would be the first 200 frames 
        #while [300:None] would be from frame 301 until the end)
        
        self.ch_width_annapurna = 640 #The width of the channels in pixels. 
        self.x_coords_annapurna = { # Define start and end X coordinate for each channel. 
            '638nm': (18, 18 + self.ch_width_annapurna),
            '561nm': (691, 691 + self.ch_width_annapurna),
            '488nm': (1370, 1370 + self.ch_width_annapurna),
            '405nm': (1370, 1370 + self.ch_width_annapurna),
        }
        
        self.ch_width_K2 = 560 #The width of the channels in pixels. 
        self.x_coords_K2 = { # Define start and end X coordinate for each channel.
            '638nm': (60, 60 + self.ch_width_K2),
            '561nm': (740, 740 + self.ch_width_K2),
            '488nm': (1415, 1415 + self.ch_width_K2),
            '405nm': (1415, 1415 + self.ch_width_K2),
        }
       
        #####USED ONLY FOR AFFINE TRANSFORM OF TIFF FILES#####
        self.channels = ['m_ch', 'l_ch'] # list with: l_ch = left, m_ch = middle, r_ch=right
        self.microscope = 'K2'  #or ANNAPURNA
        #ANNAPURNA
        self.ch_width_annapurna_tiff = 640   #The width of the channels in pixels. 
        self.x_coords_annapurna_tiff = {     # Define start and end site for each channel. 
            'l_ch': (18, 18 + self.ch_width_annapurna),  
            'm_ch': (691, 691 + self.ch_width_annapurna),  
            'r_ch': (1370, 1370 + self.ch_width_annapurna)
        }  
        
        #K2
        self.ch_width_K2_tiff = 560   
        self.x_coords_K2_tiff = { #K2
            'l_ch': (60, 60 + self.ch_width_K2),  
            'm_ch': (740, 740 + self.ch_width_K2),  
            'r_ch': (1415, 1415 + self.ch_width_K2)
        }
        ######################################################
    
class LocalizationSettings:
    def __init__(self):
        self.box = 7 #same as in picasso localize
        self.gradient405 = 200 #same as in picasso localize
        self.gradient488 = 500 #same as in picasso localize
        self.gradient561 = 200 #same as in picasso localize
        self.gradient638 = 500 #same as in picasso localize
        
        #####USED ONLY FOR AFFINE TRANSFORM OF TIFF FILES#####
        self.gradient_l = 500    #same as in picasso localize
        self.gradient_m = 500   #same as in picasso localize 
        self.gradient_r = 350   #same as in picasso localize
        ######################################################
        
        #'com' for stuff that moves too fast and does not look like a gaussian spot, 'lq' for gaussian spots. 
        self.fit_method = 'lq' 
        self.camera_info = {
            'Gain': 1,   #Set the gain of the microscope. 
            'Baseline': 100, #Baseline is the average dark camera count
            'Sensitivity': 0.6,  #Sensitivity is the conversion factor (electrons per analog-to-digital (A/D) count)
            'qe': 0.9  #In Picasso qe (quantum efficiency) is not used anymore. It is left for bacward compatibility. 
        }
        
        self.skip = ''
        self.suffix = '' #sufix for the name of the file, if necessary.
        self.transform = False #Do non-affine corrections of the localized spots if you have multiple channels.
        self.plot = True 

class LinkSettings:
    def __init__(self):
        self.coloc = False #Bool --> are you using a colocalization file or not? 
        self.dt = None  #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 
        self.quick = False #use the quick version? 200px squared from the center and only 500 frames. 
        self.roi = True #Do you want to filter per ROI? 
        self.suffix = '' #sufix for the name of the file, if necessary. 
        self.fil_len = 20 #filter the length of the tracks. Tracks shorter than this ammount of frames will be filtered
        self.fil_diff = 0.0002 #Filter for immobile particles. Tracks with a diffusion coefficient smaller than this number will be filtered
        self.tracker = 'trackpy' #Tracker algorithm to use: trackpy or swift. After talking with Chris, swift is very complicated and the focus of the developers is not 
        # really tracking, but diffusion rates. So, swift is not implemented, and I am not sure if it will. 
        self.memory = 1 #max number of frames from which a particle can disappear 
        self.search = 15#max search range for trackpy linking in px 

class ColocTracksSettings:
    def __init__(self):
        self.ch0 = '638' #reference channel
        self.ch1 = '488' #to compare with ch0 
        self.th = 250 #maximum distanc considered
        self.min_overlapped_frames = 5 #minimum amunt of frames in which the spots of the tracks have to be closer than the threshold distance set in th in a row.
        self.min_len_track = 5  #minumum length of a track to be considered for colocalization
        self.suffix = '' #sufix for the name of the file, if necessary.
        self.dt = None #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 

class ColocLocsSettings:
    def __init__(self):
        self.ch0 = '638'#'561nm'  
        self.ch1 = '488' #'638nm' 
        
        #####USED ONLY FOR AFFINE TRANSFORM OF TIFF FILES#####
        self.ch0_tiffs = 'l_ch'#'561nm'  
        self.ch1_tiffs = 'm_ch' #'638nm'  
        ######################################################
        
        self.th = 250 #Threshold distance to consider colocalization in nm. Default by Chris: 250
        self.suffix = '' #sufix for the name of the file, if necessary. 
        self.dt = None #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 
        
settings = settings.Settings(RegistrationSettings, LocalizationSettings, LinkSettings, ColocTracksSettings, ColocLocsSettings)
#%% STEP1: Affine_correct the images
if __name__ == "__main__":
    test4 = SPIT_Dataset(r'D:\Data\Chi_data\20250801_filtered', settings)
    test4.affine_transform()
    
#%%STEP 2: Go to imageJ and make manual ROIs with the freehand tool based on the maximum projection image.
#%%STEP3: localize the spots using Picasso, filter spots to be inside manually drawn ROIs, link the spots 
# to make tracks and colocalize those tracks.
if __name__ == "__main__":
    test4 = SPIT_Dataset(r'D:\Data\Chi_data\20250801_filtered', settings)
    test4.SPIT_ROI()