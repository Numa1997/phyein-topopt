import os, sys, json, time, platform, subprocess, datetime, hashlib
from pathlib import Path

T0 = time.time()
STARTED_UTC = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
WORK = Path(os.environ.get("NB_WORK", "/kaggle/working"))   # Kaggle keeps this folder as the notebook's output
SRC = Path(os.environ.get("NB_SRC", str(Path.home() / "src" / "{{PROJECT}}")))   # code, not kept as output
WORK.mkdir(parents=True, exist_ok=True)
NEED_GPU = {{NEED_GPU}}
TIMES = {}          # wall time of every stage, seconds
CHECKS = {}         # verification results, filled in section 7


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def log(msg):
    line = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}  {time.time() - T0:8.1f} s  {msg}"
    print(line, flush=True)
    with open(WORK / "progress.log", "a") as f:
        f.write(line + "\n")


class stage:
    """with stage("name"): ... times a block, logs its start and end, and records it in TIMES."""
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t = time.time(); log(f"start {self.name}"); return self
    def __exit__(self, *exc):
        TIMES[self.name] = round(time.time() - self.t, 3); log(f"end   {self.name}  ({TIMES[self.name]:.1f} s)")


def meminfo_gb(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return round(int(line.split()[1]) / 2**20, 2)


def cgroup_limit_gb():
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            return None if v == "max" or int(v) > 2**60 else round(int(v) / 2**30, 2)
        except (OSError, ValueError):
            pass


cpu_model = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None)
ENV = dict(
    started_utc=STARTED_UTC,
    on_kaggle=bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
    kaggle_run_type=os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    os=platform.platform(),
    python=sys.version.split()[0],
    cpu_model=cpu_model,
    nproc=os.cpu_count(),
    cpus_usable=len(os.sched_getaffinity(0)),
    ram_total_gb=meminfo_gb("MemTotal"),
    ram_available_gb=meminfo_gb("MemAvailable"),
    ram_cgroup_limit_gb=cgroup_limit_gb(),
    disk_free_work_gb=round(os.statvfs(WORK).f_bavail * os.statvfs(WORK).f_frsize / 2**30, 1),
    gpu=sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
)
for k, v in ENV.items():
    print(f"{k:22s} {v}")
if NEED_GPU and not ENV["gpu"]:
    raise RuntimeError("A GPU was requested but nvidia-smi finds none: stopping before any work is done.")
log("environment checked")
