from cProfile import label
from typing import Iterable, Iterator

import math
import torch as th
import numpy as np
from sklearn import datasets

class CircelsDataset(th.utils.data.IterableDataset):

    def __init__(self, start : int = 0, end : int = 10000) -> None:
        super().__init__()
        n_samples = end-start
        points, labels = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
        self.start = start
        self.end = end
        self.points = points.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __iter__(self) -> Iterator:
        worker_info = th.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return zip(self.points[iter_start:iter_end], self.labels[iter_start:iter_end])