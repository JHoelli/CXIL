import numpy as np
import timesynth as ts

def generateNewSample(dataGenerationProcess, sampler="irregular", NumTimeSteps=50, NumFeatures=50):
    '''
    Generates a new sample.
    Attributes: 
        dataGenerationProcess str: Type of Generation Process used 
        samples str: Type of sampler irregular or regular 
        NumTimeSteps int: Number Time Steps 
        Num Features int: Number of Features 
    Returns: 
        np.array: new sample (feature, time)
    '''
    dhasNoise=True

    time_sampler = ts.TimeSampler(stop_time=20)
    sample=np.zeros([NumTimeSteps,NumFeatures])


    if(sampler=="regular"):
        # keep_percentage=50 exists only for irregular NumTimeSteps*2
        time = time_sampler.sample_regular_time(num_points=NumTimeSteps)
    else:
        time = time_sampler.sample_irregular_time(num_points=NumTimeSteps*2, keep_percentage=50)

    signal= None
    for  i in range(NumFeatures):
        if(dataGenerationProcess== "Harmonic"):
                signal = ts.signals.Sinusoidal(frequency=2.0)
                
        elif(dataGenerationProcess=="GaussianProcess"):
            signal = ts.signals.GaussianProcess(kernel="Matern", nu=3./2)

        elif(dataGenerationProcess=="PseudoPeriodic"):
            signal = ts.signals.PseudoPeriodic(frequency=2.0, freqSD=0.01, ampSD=0.5)

        elif(dataGenerationProcess=="AutoRegressive"):
            signal = ts.signals.AutoRegressive(ar_param=[0.9])

        elif(dataGenerationProcess=="CAR"):
            signal = ts.signals.CAR(ar_param=0.9, sigma=0.01)

        elif(dataGenerationProcess=="NARMA"):
            signal = ts.signals.NARMA(order=10)
        else: 
            sample=np.random.normal(0,1,[NumTimeSteps,NumFeatures])

        if signal is not None:
            if(dhasNoise):
                noise= ts.noise.GaussianNoise(std=0.3)
                timeseries = ts.TimeSeries(signal, noise_generator=noise)
            else:
                timeseries = ts.TimeSeries(signal)

            feature, signals, errors = timeseries.sample(time)
            sample[:,i]= feature
    return sample