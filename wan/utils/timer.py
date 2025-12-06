import time
import numpy as np
class Timer:
    """
    This class aims to measure how much time does a set of operations take. Then generate a report on it
    """

    def __init__(self):
        self.history = {}
        self.being_tracked = set()
        pass

    def start(self, name):
        start_time = time.time()
        self.being_tracked.add(name)
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(start_time)
        return True
    
    def end(self, name):
        end_time = time.time()
        if name not in self.being_tracked:
            raise Exception(f"Wrong Timer usage. First you need to call start with function name. Called end for {name}, current being tracked: {self.being_tracked}")
        self.being_tracked.remove(name)
        self.history[name][-1] = end_time - self.history[name][-1]
        return 
    
    def summarize(self):
        avgs, vars = [], []
        for k,v in self.history.items():
            avgs.append(np.mean(v))
            vars.append(np.var(v))
        return {"values":self.history, "avgs":avgs, "vars":vars}
