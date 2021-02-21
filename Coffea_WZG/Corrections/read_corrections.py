from coffea.util import load, save


corrections = load('corrections.coffea')
print(corrections)


year="2018"

get_ele_reco_sf         = corrections['get_ele_reco_sf'][year]
get_ele_loose_id_sf     = corrections['get_ele_loose_id_sf'][year]

print(get_ele_reco_sf)
