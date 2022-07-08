import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor, hist
import numpy as np
import matplotlib.pyplot as plt
from coffea import lumi_tools
import glob
from coffea.util import load, save
from test_wrapper import JW_Processor
from pathlib import Path


if __name__ == "__main__":

    file_list = glob.glob(
        "/x6/cms/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/*.root"
    )
    year = "2018"
    sample_name = "DY"
    samples = {sample_name: file_list}
    # Class tio object
    JW_Processor_instance = JW_Processor(year, sample_name)

    ## -->Multi-node Executor
    result = processor.run_uproot_job(
        samples,  # dataset
        "Events",  # Tree name
        JW_Processor_instance,  # Class
        executor=processor.futures_executor,
        executor_args={"schema": NanoAODSchema, "workers": 48},  # number of CPU
    )

    output_directory = "output"
    Path(output_directory).mkdir(exist_ok=True, parents=True)

    outname = output_directory + "/" + sample_name + ".futures"
    save(result, outname)
