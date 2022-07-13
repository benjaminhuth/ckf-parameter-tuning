 #!/usr/bin/env python3
import sys
import os
import yaml
import pprint
import time
import warnings

from typing import Optional, Union
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
import acts
import acts.examples

u = acts.UnitConstants

from common import getOpenDataDetector, getOpenDataDetectorDirectory
from particle_gun import addParticleGun, EtaConfig, ParticleConfig, MomentumConfig
from pythia8 import addPythia8
from digitization import addDigitization
from fatras import addFatras
from exatrkx import addExaTrkx
from seeding import addSeeding, SeedingAlgorithm, TruthSeedRanges, TrackParamsEstimationConfig, SeedfinderConfigArg
from ckf_tracks import addCKFTracks, CKFPerformanceConfig



def run_ckf(trackingGeometry,
            outputFile,
            seedfinderConfig = SeedfinderConfigArg()
):


    ######################
    # Check config files #
    ######################

    oddDir = getOpenDataDetectorDirectory()

    oddDigiConfigSmear = oddDir / "config/odd-digi-smearing-config.json"
    assert os.path.exists(oddDigiConfigSmear)

    oddGeoSelectionSeeding = oddDir / "config/odd-material-mapping-config.json" # ?????????????????
    assert os.path.exists(oddGeoSelectionSeeding)

        
    #############################
    # Prepare and run sequencer #
    #############################

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
    
    outputDir = Path("output")
    if not outputDir.exists():
        outputDir.mkdir(exist_ok=True, parents=True)

    s = acts.examples.Sequencer(
        events=9,
        numThreads=3,
        logLevel=acts.logging.FATAL,
        outputDir = str(outputDir),
        outputTimingFile = "timing-{}.tsv".format(os.getpid()),
    )

    rnd_seed = int(time.time())
    rnd = acts.examples.RandomNumbers(seed=rnd_seed)

    #s = addParticleGun(
        #s,
        #MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, True),
        #EtaConfig(-3.0, 3.0, True),
        #ParticleConfig(1, acts.PdgParticle.eMuon, True),
        #rnd=rnd,
        #multiplicity=50,
    #)

    s = addPythia8(
        s,
        rnd=rnd,
        hardProcess=["HardQCD:all = on"],
        #hardProcess=["Top:qqbar2ttbar=on"],
    )
            
    s.addAlgorithm(
        acts.examples.ParticleSelector(
            level=s.config.logLevel,
            inputParticles="particles_input",
            outputParticles="particles_selected",
            removeNeutral=True,
            absEtaMax=3,
            rhoMax=2.0 * u.mm,
            ptMin=500 * u.MeV,
        )
    )

    s.addAlgorithm(
        acts.examples.FatrasSimulation(
            level=s.config.logLevel,
            inputParticles="particles_selected",
            outputParticlesInitial="particles_initial",
            outputParticlesFinal="particles_final",
            outputSimHits="simhits",
            randomNumbers=rnd,
            trackingGeometry=trackingGeometry,
            magneticField=field,
            generateHitsOnSensitive=True,
        )
    )

    s = addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=oddDigiConfigSmear,
        outputDirRoot=None,
        rnd=rnd,
    )

    s = addSeeding(
        s,
        trackingGeometry,
        field,
        seedfinderConfigArg=seedfinderConfig,
        geoSelectionConfigFile=oddGeoSelectionSeeding,
        initialVarInflation=[100, 100, 100, 100, 100, 100],
        seedingAlgorithm=SeedingAlgorithm.Default,
    )

    s.addAlgorithm(
        acts.examples.TrackFindingAlgorithm(
            level=acts.logging.FATAL,
            measurementSelectorCfg=acts.MeasurementSelector.Config(
                [(acts.GeometryIdentifier(), ([], [15.0], [10]))]
            ),
            inputMeasurements="measurements",
            inputSourceLinks="sourcelinks",
            inputInitialTrackParameters="estimatedparameters",
            outputTrajectories="ckf_trajectories",
            findTracks=acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction(
                trackingGeometry, field
            ),
        )
    )

    s.addAlgorithm(
        acts.examples.TrajectoriesToPrototracks(
            level=acts.logging.FATAL,
            inputTrajectories="ckf_trajectories",
            outputPrototracks="ckf_prototracks",
        )
    )
        
    s.addWriter(
        acts.examples.CKFPerformanceWriter(
            level=acts.logging.FATAL,
            inputTrajectories = "ckf_trajectories",
            inputParticles = "particles_initial",
            inputMeasurementParticlesMap = "measurement_particles_map",
            filePath = str(outputFile),
            truthMatchProbMin = 0.5,
            nMeasurementsMin = 9,
            ptMin=500 * u.MeV,
        )
    )
            
    s.run() 

if __name__ == "__main__":
    oddDir = getOpenDataDetectorDirectory()

    oddMaterialMap = oddDir / "data/odd-material-maps.root"
    assert os.path.exists(oddMaterialMap)

    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    _, trackingGeometry, _ = getOpenDataDetector(mdecorator=oddMaterialDeco)
    
    run_ckf(trackingGeometry, "bla.root")
