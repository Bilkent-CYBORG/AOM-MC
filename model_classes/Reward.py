"""
This class represents the outcome of an arm (i.e. performance as stated in the paper)
"""


class Reward:
    def __init__(self, worker, performance, temp_perf=None, t=1):
        self.temp_perf = temp_perf
        self.t = t
        self.performance = performance
        self.context = worker.context
        self.worker = worker
