from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import numpy as np
import uproot3
from coffea.util import save, load
from coffea import hist, lookup_tools
from coffea.lookup_tools import extractor, dense_lookup


###
# Electron trigger efficiency SFs. depends on supercluster eta and pt:
###

ele_trig_hists_leg1 = {
    '2018': uproot3.open("Trigger/leg1/egammaEffi.txt_EGM2D.root")
}

get_ele_trig_leg1_SF = {}
get_ele_trig_leg1_data_Eff = {}
get_ele_trig_leg1_mc_Eff = {}
for year in ['2018']:
    ele_trig_SF_leg1 = ele_trig_hists_leg1[year]['EGamma_SF2D']
    ele_trig_Eff_data_leg1 = ele_trig_hists_leg1[year]['EGamma_EffData2D']
    ele_trig_Eff_mc_leg1 = ele_trig_hists_leg1[year]['EGamma_EffMC2D']

    get_ele_trig_leg1_SF[year]		 = lookup_tools.dense_lookup.dense_lookup(ele_trig_SF_leg1.values, ele_trig_SF_leg1.edges)
    get_ele_trig_leg1_data_Eff[year] = lookup_tools.dense_lookup.dense_lookup(ele_trig_Eff_data_leg1.values, ele_trig_Eff_data_leg1.edges)
    get_ele_trig_leg1_mc_Eff[year]	 = lookup_tools.dense_lookup.dense_lookup(ele_trig_Eff_mc_leg1.values, ele_trig_Eff_mc_leg1.edges)


ele_trig_hists_leg2 = {
    '2018': uproot3.open("Trigger/leg2/egammaEffi.txt_EGM2D.root")
}
get_ele_trig_leg2_SF = {}
get_ele_trig_leg2_data_Eff = {}
get_ele_trig_leg2_mc_Eff = {}
for year in ['2018']:
    ele_trig_SF_leg2 = ele_trig_hists_leg2[year]['EGamma_SF2D']
    ele_trig_Eff_data_leg2 = ele_trig_hists_leg2[year]['EGamma_EffData2D']
    ele_trig_Eff_mc_leg2 = ele_trig_hists_leg2[year]['EGamma_EffMC2D']

    get_ele_trig_leg2_SF[year]		 = lookup_tools.dense_lookup.dense_lookup(ele_trig_SF_leg2.values, ele_trig_SF_leg2.edges)
    get_ele_trig_leg2_data_Eff[year] = lookup_tools.dense_lookup.dense_lookup(ele_trig_Eff_data_leg2.values, ele_trig_Eff_data_leg2.edges)
    get_ele_trig_leg2_mc_Eff[year]	 = lookup_tools.dense_lookup.dense_lookup(ele_trig_Eff_mc_leg2.values, ele_trig_Eff_mc_leg2.edges)


###
# Electron id SFs. depends on supercluster eta and pt.
###

ele_medium_files = {
    '2018': uproot3.open("Egamma/Electron_hist_root/2018/egammaEffi.txt_Ele_Medium_EGM2D.root"),
    '2017': uproot3.open("Egamma/Electron_hist_root/2017/egammaEffi.txt_EGM2D_Medium_UL17.root"),
    '2016_preVFP': uproot3.open("Egamma/Electron_hist_root/2016/preVFP/egammaEffi.txt_Ele_Medium_preVFP_EGM2D.root"),
    '2016_postVFP': uproot3.open("Egamma/Electron_hist_root/2016/postVFP/egammaEffi.txt_Ele_Medium_postVFP_EGM2D.root")
}

get_ele_medium_id_sf = {}
for year in ['2018','2017','2016_preVFP','2016_postVFP']:
    ele_medium_sf_hist = ele_medium_files[year]["EGamma_SF2D"]
    get_ele_medium_id_sf[year]  = lookup_tools.dense_lookup.dense_lookup(ele_medium_sf_hist.values, ele_medium_sf_hist.edges)


###
# Photon id SFs. depends on supercluster eta and pt.
###


pho_medium_files = {
    '2018': uproot3.open("Egamma/Photon_hist_root/2018/egammaEffi.txt_EGM2D_Pho_Med_UL18.root"),
    '2017': uproot3.open("Egamma/Photon_hist_root/2017/egammaEffi.txt_EGM2D_PHO_Medium_UL17.root"),
    '2016_preVFP': uproot3.open("Egamma/Photon_hist_root/2016/preVFP/egammaEffi.txt_EGM2D_Pho_Medium_UL16.root"),
    '2016_postVFP': uproot3.open("Egamma/Photon_hist_root/2016/postVFP/egammaEffi.txt_EGM2D_Pho_Medium_UL16.root")
}

get_pho_medium_id_sf = {}
for year in ['2018','2017','2016_preVFP','2016_postVFP']:
    pho_medium_sf_hist = pho_medium_files[year]["EGamma_SF2D"]
    get_pho_medium_id_sf[year]  = lookup_tools.dense_lookup.dense_lookup(pho_medium_sf_hist.values, pho_medium_sf_hist.edges)




###
# Electron reconstruction SFs. Depends on supercluster eta and pt.    
###

ele_reco_above20_files = {
    '2018': uproot3.open("Egamma/Electron_hist_root/2018/egammaEffi_ptAbove20.txt_EGM2D_UL2018.root"),
    '2017': uproot3.open("Egamma/Electron_hist_root/2017/egammaEffi_ptAbove20.txt_EGM2D_UL2017.root"),
    '2016_preVFP': uproot3.open("Egamma/Electron_hist_root/2016/preVFP/egammaEffi_ptAbove20.txt_EGM2D_UL2016preVFP.root"),
    '2016_postVFP': uproot3.open("Egamma/Electron_hist_root/2016/postVFP/egammaEffi_ptAbove20.txt_EGM2D_UL2016postVFP.root")
}
get_ele_reco_above20_sf = {}
for year in ['2018','2017','2016_preVFP','2016_postVFP']:
    ele_reco_hist = ele_reco_above20_files[year]["EGamma_SF2D"]
    get_ele_reco_above20_sf[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.values, ele_reco_hist.edges)


ele_reco_below20_files = {
    '2018': uproot3.open("Egamma/Electron_hist_root/2018/egammaEffi_ptBelow20.txt_EGM2D_UL2018.root"),
    '2017': uproot3.open("Egamma/Electron_hist_root/2017/egammaEffi_ptBelow20.txt_EGM2D_UL2017.root"),
    '2016_preVFP': uproot3.open("Egamma/Electron_hist_root/2016/preVFP/egammaEffi_ptBelow20.txt_EGM2D_UL2016preVFP.root"),
    '2016_postVFP': uproot3.open("Egamma/Electron_hist_root/2016/postVFP/egammaEffi_ptBelow20.txt_EGM2D_UL2016postVFP.root")
}

get_ele_reco_below20_sf = {}
for year in ['2018','2017','2016_preVFP','2016_postVFP']:
    ele_reco_hist = ele_reco_below20_files[year]["EGamma_SF2D"]
    get_ele_reco_below20_sf[year]=lookup_tools.dense_lookup.dense_lookup(ele_reco_hist.values, ele_reco_hist.edges)


corrections = {   
#    'get_pu_weight' : get_pu_weight,
'get_ele_medium_id_sf':get_ele_medium_id_sf,
'get_pho_medium_id_sf':get_pho_medium_id_sf,
'get_ele_reco_above20_sf':get_ele_reco_above20_sf,
'get_ele_reco_below20_sf':get_ele_reco_below20_sf,
'get_ele_trig_leg1_SF':get_ele_trig_leg1_SF,
'get_ele_trig_leg1_data_Eff':get_ele_trig_leg1_data_Eff,
'get_ele_trig_leg1_mc_Eff':get_ele_trig_leg1_mc_Eff,
'get_ele_trig_leg2_SF':get_ele_trig_leg2_SF,
'get_ele_trig_leg2_data_Eff':get_ele_trig_leg2_data_Eff,
'get_ele_trig_leg2_mc_Eff':get_ele_trig_leg2_mc_Eff
}


save(corrections,'corrections.coffea')
