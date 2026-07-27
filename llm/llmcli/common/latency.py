import time
import torch

class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t1 = time.perf_counter()

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0
