import os
import sys
import pprint
import json
import time
import datetime
from pathlib import Path
from multiprocessing import Pool

import optuna
import uproot
import pandas as pd
import matplotlib.pyplot as plt

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from ckf import run_ckf

import acts
from common import getOpenDataDetector, getOpenDataDetectorDirectory


from seeding import SeedfinderConfigArg


class Objective:
    def __init__(self, k_dup, k_time):
        self.res = {
            "eff": [],
            "fakerate": [],
            "duplicaterate": [],
            "runtime": [],
        }
        
        self.k_dup = k_dup
        self.k_time = k_time
        
        oddDir = getOpenDataDetectorDirectory()

        oddMaterialMap = oddDir / "data/odd-material-maps.root"
        assert os.path.exists(oddMaterialMap)

        oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
        _, self.trackingGeometry, _ = getOpenDataDetector(mdecorator=oddMaterialDeco)

    def __call__(self, trial):
        #maxPtScattering = trial.suggest_float("maxPtScattering", -10, 10) do not use
        impactMax = trial.suggest_float("impactMax", 0.1, 25)
        deltaRMin = trial.suggest_float("deltaRMin", 0.25, 30)
        deltaRMax = trial.suggest_float("deltaRMax", 50, 300)
        sigmaScattering = trial.suggest_float("sigmaScattering", 0.2, 50)
        radLengthPerSeed = trial.suggest_float("radLengthPerSeed", 0.001, 0.1)
        maxSeedsPerSpM = trial.suggest_int("maxSeedsPerSpM", 0, 5)
        cotThetaMax = trial.suggest_float("cotThetaMax", 5, 20)
        
        seedfinderConfig = SeedfinderConfigArg(
            cotThetaMax=cotThetaMax,
            sigmaScattering=sigmaScattering,
            impactMax=impactMax,
            deltaR=(deltaRMin, deltaRMax),
            radLengthPerSeed=radLengthPerSeed,
            maxSeedsPerSpM=maxSeedsPerSpM
        )
        
        outputFile = "output/performance-{}.root".format(os.getpid())
        run_ckf(self.trackingGeometry, outputFile, seedfinderConfig)
        
        rootFile = uproot.open(outputFile)
        self.res["eff"].append(rootFile["eff_particles"].member("fElements")[0])
        self.res["fakerate"].append(rootFile["fakerate_particles"].member("fElements")[0])
        self.res["duplicaterate"].append(rootFile["duplicaterate_particles"].member("fElements")[0])
                
        os.remove(outputFile)

        timing = pd.read_csv("output/timing-{}.tsv".format(os.getpid()), sep="\t")
        time_ckf = float(timing[timing['identifier'].str.match("Algorithm:TrackFindingAlgorithm")]['time_perevent_s'])
        time_seeding = float(timing[timing['identifier'].str.match("Algorithm:SeedingAlgorithm")]['time_perevent_s'])
        self.res["runtime"].append(time_ckf + time_seeding)
        
        pprint.pprint({ key: self.res[key][-1] for key in self.res.keys() }, indent=4)
        
        return self.res["eff"][-1] - (self.res["fakerate"][-1] + 
                                      self.res["duplicaterate"][-1]/self.k_dup + 
                                      (self.res["runtime"][-1]**2)/self.k_time)



def run_study(study_name, n_trials, k_dup, k_time, start_values, process_index):
    print("Start process {} with pid {}".format(process_index, os.getpid()))

    objective = Objective(k_dup, k_time)

    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///{}.db".format(study_name),
        direction='maximize',
        load_if_exists=True
    )
    
    
    study.enqueue_trial(start_values)

    study.optimize(objective, n_trials=n_trials)
    
    return objective.res
    

try:
    study_name = sys.argv[1]
    n_trials = int(sys.argv[2])
    n_processes = int(sys.argv[3])
    k_dup = int(sys.argv[4])
    k_time = int(sys.argv[5])
except:
    print("Usage: {} <name> <n_trials> <n_processes> <k_dup> <k_time>".format(sys.argv[0]))
    exit()
    

start_values = {
    "cotThetaMax": 16.541921673890172,
    "deltaRMax": 50.0854850448914,
    "deltaRMin": 13.639924973033985,
    "impactMax": 4.426123855748383,
    "maxSeedsPerSpM": 0,
    "radLengthPerSeed": 0.06311548593790932,
    "sigmaScattering": 7.3401486140533985
}

with Pool(n_processes) as p:
    results = []
    
    for i in range(n_processes):
        res = p.apply_async(run_study, args = (study_name, n_trials, k_dup, k_time, start_values, i))
        time.sleep(0.1)
        results.append(res)
        
    run_study(study_name, n_trials, k_dup, k_time, start_values, -1)
    
    [res.wait() for res in results]
    

study = optuna.create_study(
    study_name=study_name,
    storage="sqlite:///{}.db".format(study_name),
    direction='maximize',
    load_if_exists=True
)

outputDir = Path(study.study_name)
outputDir.mkdir(exist_ok=True)

print(study.best_params)

with open(outputDir / 'result.json', 'w') as f:
    json.dump(study.best_params, f)

#fig, axes = plt.subplots(2,2, figsize=(15,12))

#for key, ax in zip(results[0].keys(), axes.flatten()):
    #ax.plot(objective.res[key])
    #ax.set_title(key)
    
#plt.tight_layout()
#plt.show()

#fig.savefig(outputDir / 'plots.pdf')

fig_hist = plot_optimization_history(study)
fig_hist.write_image(outputDir / "opt_history.jpeg")

all_params = ["impactMax","deltaRMin","sigmaScattering","deltaRMax","maxSeedsPerSpM","radLengthPerSeed","cotThetaMax"]
fig_parallel = plot_parallel_coordinate(study,params=all_params)
fig_parallel.write_image(outputDir / "opt_parallel.jpeg")
