import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# Read tree and show branches
infile = "../data/W_enu_pu200.root:Delphes"  # tree name: Delphes
tree = uproot.open(infile)

# Show the number of events
print("Number of events: {0}".format(tree.num_entries))

# Save the cut-flow
cut0 = np.zeros(tree.num_entries)

# Branch to array

tree_arr = tree.arrays(filter_name=["Electron*", "MissingET*"])
# You can add other particels (e.g. filter_name=['Jet*]')

# Electron group ( for more convenient )
Electron = ak.zip(
    {
        "PT": tree_arr["Electron.PT"],
        "Eta": tree_arr["Electron.Eta"],
        "Phi": tree_arr["Electron.Phi"],
        "T": tree_arr["Electron.T"],
        "Charge": tree_arr["Electron.Charge"],
    }
)

# MET group
MET = ak.zip(
    {
        "MET": tree_arr["MissingET.MET"],
        "Eta": tree_arr["MissingET.Eta"],
        "Phi": tree_arr["MissingET.Phi"],
    }
)

# Minimum PT cut >= 10 GeV
Ele_PT_mask = Electron.PT >= 10

# Tracker coverage |eta| <= 2.5
Ele_Eta_mask = abs(Electron.Eta) <= 2.5

# Combine cut1 and cut2 using AND(&) operator
Electron_selection_mask = (Ele_PT_mask) & (Ele_Eta_mask)

# Event mask
Electron = Electron[Electron_selection_mask]
Electron_event_mask = ak.num(Electron) >= 1

# Apply event selection
Electron = Electron[Electron_event_mask]
MET = MET[Electron_event_mask]
# Photon = Photo[Electron_event_mask]
# Jet = Jet[Electron_event_mask]


# CutFlow
cut1 = np.ones(len(Electron))


print("Cut0: {0}, Cut1: {1}".format(len(cut0), len(cut1)))


import vector

leading_Electron = Electron[:, 0]  # Highest PT
Ele2vec = vector.obj(pt=leading_Electron.PT, phi=leading_Electron.Phi)
MET2vec = vector.obj(pt=MET.MET, phi=MET.Phi)
MT = np.sqrt(
    2 * leading_Electron.PT * MET.MET * (1 - np.cos(abs(MET2vec.deltaphi(Ele2vec))))
)


def draw(arr, title, start, end, bin):  # Array, Name, x_min, x_max, bin-number
    plt.figure(figsize=(8, 5))  # Figure size
    bins = np.linspace(start, end, bin)  # divide start-end range with 'bin' number
    binwidth = (end - start) / bin  # width of one bin

    # Draw histogram
    plt.hist(arr, bins=bins, alpha=0.7, label=title)  # label is needed to draw legend

    plt.xticks(fontsize=16)  # xtick size
    plt.xlabel(title, fontsize=16)  # X-label
    # plt.xlabel('$e_{PT}',fontsize=16) # X-label (If you want LateX format)

    plt.ylabel("Number of Events/(%d GeV)" % binwidth, fontsize=16)  # Y-label
    # plt.ylabel('Number of Events',fontsize=16) # Y-label withou bin-width
    plt.yticks(fontsize=16)  # ytick size

    plt.grid(alpha=0.5)  # grid
    plt.legend(prop={"size": 15})  # show legend
    # plt.yscale('log')	# log scale

    outname_fig = title + ".png"
    plt.savefig(outname_fig)
    plt.show()  # show histogram
    plt.close()


# Electron PT
draw(ak.flatten(Electron.PT), "Electron_PT", 0, 700, 25)

# MET PT
draw(ak.flatten(MET.MET), "MET_PT", 0, 800, 25)

# MT
draw(ak.flatten(MT), "W_MT", 0, 1400, 25)
