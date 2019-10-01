import numpy as np, scipy.stats as st


def confidence_interval(a, confidence=0.95):
    return st.t.interval(confidence, len(a)-1, loc=np.mean(a), scale=st.sem(a))
