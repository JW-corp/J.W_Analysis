## Template fit method for Fake-Photon estimation study  

### Pre-requisite
The fitting code use the input as "npy" file which has information about bin and bin-contents of histogram
Before the start, we need make this "npy" file from "coffea-processor" output
 - [preprocess](https://github.com/JW-corp/J.W_Analysis/blob/main/Coffea_WZG/Condor_coffea/N04_draw_fake.py)



### Files 

 - [Fit_root.py](https://github.com/JW-corp/J.W_Analysis/blob/main/Coffea_WZG/Fitting/Fit_root.py): Fit using TFractionFitter in ROOT (stable)
 - [Fit_pyhf.py](https://github.com/JW-corp/J.W_Analysis/blob/main/Coffea_WZG/Fitting/Fit_pyhf.py): Fit using Fully python eco-system tool: pyhf (need improve)  
 - [root_set](https://github.com/JW-corp/J.W_Analysis/blob/main/Coffea_WZG/Fitting/root_set): Setup root ( python3.8 + ROOT6 ) in KNU-CCP cluster.  
 - images
   - images_2017 : Fit results with 2017 data
   - images_2018 : Fit results with 2018 data
   - images_2018_UncSB_sample1 : Fit results with 2018 IsoChg Sideband Unc study sample1
   - images_2018_UncSB_sample2 : Fit results with 2018 IsoChg Sideband Unc study sample2


---
### Comments
The fit-method with pyhf need improvement in this study.
It shows low performance. But I think it can show high performance after I study this more in the future.
