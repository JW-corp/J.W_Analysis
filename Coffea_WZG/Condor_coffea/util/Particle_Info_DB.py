GenDict_2017 = {
    "WZG": 128000,
    "WZ": 7898000,
    "ZZ": 2000000,
    "TTWJets": 5216439,
    "TTZtoLL": 14000000,
    "tZq": 13860200,
    "ZGToLLG": 29664676,
    "TTGJets": 4970959,
}

xsecDict_2017 = {
    "WZG": 0.0196,
    "WZ": 27.6,
    "ZZ": 12.14,
    "TTWJets": 0.2149,
    "TTZtoLL": 0.2432,
    "tZq": 0.07358,
    "ZGToLLG": 55.48,
    "TTGJets": 4.078,
}


GenDict_2018 = {
    "WZG": 128000,
    "DY": 1933600,
    "WZ": 7986000,
    "ZZ": 2000000,
    "TTWJets": 4963867,
    "TTZtoLL": 13914900,
    "tZq": 12748300,
    "ZGToLLG": 28636926,
    "TTGJets": 4647426,
    "WGToLNuG": 20371504,
}

xsecDict_2018 = {
    "WZG": 0.0196,
    "DY": 2137.0,
    "WZ": 27.6,
    "ZZ": 12.14,
    "TTWJets": 0.2149,
    "TTZtoLL": 0.2432,
    "tZq": 0.07358,
    "ZGToLLG": 55.48,
    "TTGJets": 4.078,
    "WGToLNuG": 1.249,
}


DB = {
    "2018": {
        "Lumi": 53.03,
        "Gen": GenDict_2018,
        "xsec": xsecDict_2018,
    },
    "2017": {
        "Lumi": 41.557,
        "Gen": GenDict_2017,
        "xsec": xsecDict_2017,
    },
}
