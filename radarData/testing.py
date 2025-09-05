# add the top directory to the python system path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.radar.createRadarData import createModelData

# radar params
radarParams = {
    "pulseTime": 20e-6,
    "pulseStartFreq": 0,
    "pulseRepFreq": 5e2,
    "bandwidth": 2e6,
    "maxRange": 60e3,
    "sampleRate": 8e6,
    "targetRange" : 20e3,
    "snr": [10, 5, 0, -5, -10, "varying"],
    "maxRangeSnr" : 0,
    "targetPresent" : True,
    "noisePresent" : True,
    "noiseDist" : ["complex gaussian", "complex weibull", "complex student_t"]
    }
            
# create model data         
createModelData(radarParams=radarParams,
                nTrainData=100, 
                nTestData=10, 
                nEvalData=100,
                exampleSignal=False,
                createData=True,
                verbose=True)
