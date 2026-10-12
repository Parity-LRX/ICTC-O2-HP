"""Sample this process's NVML memory through the driver's read-only CLI."""
import os
import subprocess
import threading
import time


class ProcessMemorySampler:
    def __init__(self, interval=.05):
        self.interval = interval
        self.samples = []
        self.errors = []
        self.done = threading.Event()
        self.thread = None

    def sample(self):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"], text=True, timeout=2)
            for line in output.splitlines():
                pid, memory = line.split(",", 1)
                if int(pid.strip()) == os.getpid():
                    self.samples.append((time.monotonic(), float(memory.strip())))
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            self.errors.append(repr(exc))

    def loop(self):
        while not self.done.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.sample()
        return {"sampled_process_peak_MiB": max((z[1] for z in self.samples), default=None),
                "process_memory_samples": len(self.samples),
                "process_memory_requested_interval_s": self.interval,
                "process_memory_observed_max_interval_s": max(
                    (b[0]-a[0] for a,b in zip(self.samples, self.samples[1:])), default=None),
                "process_memory_sampling_errors": self.errors,
                "process_memory_definition": "sampled NVML process memory; includes context/cache, not an exact transient peak"}
