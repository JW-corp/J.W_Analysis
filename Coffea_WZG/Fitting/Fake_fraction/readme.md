### Fake fraction file


1. npy files are dictionary of fake fractions.

2. [open.py](https://github.com/JW-corp/J.W_Analysis/blob/main/Coffea_WZG/Fitting/Fake_fraction/open.py) can view the dictionary

3. The key of dictionary is bin name and the value means fake fraction.
```python
{
"PT_1_eta_1" : 0.2851229506158348
"PT_1_eta_2" : 0.2671860368715074
"PT_1_eta_3" : 0.19775958618458792
"PT_1_eta_4" : 0.3267804644873494
"PT_2_eta_1" : 0.1265894655999156
"PT_2_eta_2" : 0.12835109858188273
"PT_2_eta_3" : 0.12584781590326632
"PT_2_eta_4" : 0.11211688549664137
"PT_3_eta_1" : 0.05397628579707204
"PT_3_eta_2" : 0.16442854300440082
"PT_3_eta_3" : 0.11756851018550148
"PT_3_eta_4" : 0.10191349914743844
"PT_4_eta_1" : 0.031874331463749524
"PT_4_eta_2" : 0.05858789594488875
"PT_4_eta_3" : 0.0373471553147374
"PT_4_eta_4" : 0.10274145038024736
}
```
  
The bin-names and cuts are mached like this

| bin name   | cut                             |
|------------|---------------------------------|
| PT_1_eta_1 | 20 < pt <30 & \|eta\| < 1       |
| PT_1_eta_2 | 20 < pt <30 & 1 < \|eta\| < 1.5 |
| PT_1_eta_3 | 20 < pt <30 & 1.5 < \|eta\| < 2 |
| PT_1_eta_4 | 20 < pt <30 & 2 < \|eta\| < 2.5 |
| PT_2_eta_1 | 30 < pt <40 & \|eta\| < 1       |
| PT_2_eta_2 | 30 < pt <40 & 1 < \|eta\| < 1.5 |
| PT_2_eta_3 | 30 < pt <40 & 1.5 < \|eta\| < 2 |
| PT_2_eta_4 | 30 < pt <40 & 2 < \|eta\| < 2.5 |
| PT_3_eta_1 | 40 < pt <50 & \|eta\| < 1       |
| PT_3_eta_2 | 40 < pt <50 & 1 < \|eta\| < 1.5 |
| PT_3_eta_3 | 40 < pt <50 & 1.5 < \|eta\| < 2 |
| PT_3_eta_4 | 40 < pt <50 & 2 < \|eta\| < 2.5 |
| PT_4_eta_1 | 50 < pt & \|eta\| < 1           |
| PT_4_eta_2 | 50 < pt,& 1 < \|eta\| < 1.5     |
| PT_4_eta_3 | 50 < pt,& 1.5 < \|eta\| < 2     |
| PT_4_eta_4 | 50 < pt,& 2 < \|eta\| < 2.5     |



[Fake Photon array generation in my script](https://github.com/JW-corp/J.W_Analysis/blob/main/Coffea_WZG/Condor_coffea/N03_run_processor.py#L571)
  
Fake fraction should be applied on 16 different bins.
This code-snippet make fake fraction arrays matched with different Photon PT & Eta bins
After that you can make Fake Photon sample using this simple equation  

  
*Number of Fake Photon sample = fake fraction(fw) * Number of Data*
  


```python
## -- Fake-fraction Lookup table --##

# -------------------- Make Fake Photon BKGs---------------------------#
def make_bins(pt, eta, bin_range_str):

    bin_dict = {
        "PT_1_eta_1": (pt > 20) & (pt < 30) & (eta < 1),
        "PT_1_eta_2": (pt > 20) & (pt < 30) & (eta > 1) & (eta < 1.5),
        "PT_1_eta_3": (pt > 20) & (pt < 30) & (eta > 1.5) & (eta < 2),
        "PT_1_eta_4": (pt > 20) & (pt < 30) & (eta > 2) & (eta < 2.5),
        "PT_2_eta_1": (pt > 30) & (pt < 40) & (eta < 1),
        "PT_2_eta_2": (pt > 30) & (pt < 40) & (eta > 1) & (eta < 1.5),
        "PT_2_eta_3": (pt > 30) & (pt < 40) & (eta > 1.5) & (eta < 2),
        "PT_2_eta_4": (pt > 30) & (pt < 40) & (eta > 2) & (eta < 2.5),
        "PT_3_eta_1": (pt > 40) & (pt < 50) & (eta < 1),
        "PT_3_eta_2": (pt > 40) & (pt < 50) & (eta > 1) & (eta < 1.5),
        "PT_3_eta_3": (pt > 40) & (pt < 50) & (eta > 1.5) & (eta < 2),
        "PT_3_eta_4": (pt > 40) & (pt < 50) & (eta > 2) & (eta < 2.5),
        "PT_4_eta_1": (pt > 50) & (eta < 1),
        "PT_4_eta_2": (pt > 50) & (eta > 1) & (eta < 1.5),
        "PT_4_eta_3": (pt > 50) & (eta > 1.5) & (eta < 2),
        "PT_4_eta_4": (pt > 50) & (eta > 2) & (eta < 2.5),
    }

    binmask = bin_dict[bin_range_str]

    return binmask

bin_name_list = [
    "PT_1_eta_1",
    "PT_1_eta_2",
    "PT_1_eta_3",
    "PT_1_eta_4",
    "PT_2_eta_1",
    "PT_2_eta_2",
    "PT_2_eta_3",
    "PT_2_eta_4",
    "PT_3_eta_1",
    "PT_3_eta_2",
    "PT_3_eta_3",
    "PT_3_eta_4",
    "PT_4_eta_1",
    "PT_4_eta_2",
    "PT_4_eta_3",
    "PT_4_eta_4",
]


if isFake:
    # Make Bin-range mask
    binned_pteta_mask = {}
    for name in bin_name_list:
        binned_pteta_mask[name] = make_bins(
            ak.flatten(leading_pho.pt),
            ak.flatten(abs(leading_pho.eta)),
            name,
        )
    # Read Fake fraction --> Mapping bin name to int()

    if self._year == "2018":
        in_dict = np.load('Fitting_2018/Fit_results.npy',allow_pickle="True")[()]

    if self._year == "2017":
        in_dict = np.load('Fitting_2017/Fit_results.npy',allow_pickle="True")[()]

    idx=0
    fake_dict ={}
    for i,j in in_dict.items():
        fake_dict[idx] = j
        idx+=1


    # Reconstruct Fake_weight
    fw= 0
    for i,j in binned_pteta_mask.items():
        fw = fw + j*fake_dict[bin_name_list.index(i)]


    # Process 0 weight to 1
    @numba.njit
    def zero_one(x):
        if x == 0:
            x = 1
        return x
    vec_zero_one = np.vectorize(zero_one)
    fw = vec_zero_one(fw)
else:
    fw = np.ones(len(events))
```
