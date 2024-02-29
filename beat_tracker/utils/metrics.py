from mir_eval import beat
import numpy as np


def f_measure(estimated, reference, threshold=0.07):
    
    # print(f"Estimated: {estimated[0]}")
    # print(f"Reference: {reference[0]}")
    
    f_measures = []
    for i in range(len(estimated)):
        f_measure = beat.f_measure(reference[i], estimated[i], threshold)
        f_measures.append(f_measure)
        
    f_measures = np.array(f_measures)
    return f_measures.mean()
    
def continuity(estimated, reference):
    
    # cmlc,cmlt,amlc,amlt =  beat.continuity(reference_beats = reference, estimated_beats = estimated)
    
    cmlc = []
    cmlt = []
    amlc = []
    amlt = []
    
    for i in range(len(estimated)):
        cmlc_,cmlt_,amlc_,amlt_ = beat.continuity(reference_beats = reference[i], estimated_beats = estimated[i])
        cmlc.append(cmlc_)
        cmlt.append(cmlt_)
        amlc.append(amlc_)
        amlt.append(amlt_)
    
    return np.array(cmlc).mean(), np.array(cmlt).mean(), np.array(amlc).mean(), np.array(amlt).mean()