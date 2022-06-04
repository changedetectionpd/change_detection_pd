import numpy as np

class TimeDelayEmbedding:

    def __init__(self, dim=3, delay=1, skip=1):
        self._dim = dim
        self._delay = delay
        self._skip = skip

    def __call__(self, ts, n):
        if n==0:
            return self._transform(np.array(ts))
        else:
            return self.transform(np.array(ts))


    def fit(self, ts, y=None):
        return self
    
    def _transform(self, ts):
        if ts.ndim == 1:
            repeat = self._dim
        else:
            assert self._dim % ts.shape[1] == 0
            repeat = self._dim // ts.shape[1]
        end = len(ts) - self._delay * (repeat - 1)
        short = np.arange(0, end, self._skip)
        vertical = np.arange(0, repeat * self._delay, self._delay)
        return ts[np.add.outer(short, vertical)].reshape(len(short), -1)

    def transform(self, ts):
        return [self._transform(np.array(s)) for s in ts]