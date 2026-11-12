import importlib.metadata as md

REPO = "{{REPO_URL}}"
COMMIT = "{{COMMIT}}"
REQUIRES = {{REQUIRES}}   # the packages this notebook imports, as named in requirements.txt

with stage("clone"):
    if not (SRC / ".git").exists():
        SRC.mkdir(parents=True, exist_ok=True)
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", REPO],
                    ["git", "fetch", "-q", "--depth", "1", "origin", COMMIT], ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=SRC, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC, capture_output=True, text=True, check=True).stdout.strip()
    assert head == COMMIT, f"checked out {head}, expected {COMMIT}"
    log(f"code at {REPO} commit {head}")

# Pinned versions from the repository at that commit ("name==version" or "name[extra]==version").
PINS = {}
for line in (SRC / "requirements.txt").read_text().splitlines():
    line = line.split("#", 1)[0].strip()
    if "==" in line:
        spec_, ver = line.split("==", 1)
        PINS[spec_.split("[", 1)[0].strip().lower()] = (line, ver.strip())


def installed(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


with stage("install"):
    missing = [PINS[p][0] for p in REQUIRES if installed(p) is None]
    if missing:
        log("installing (missing from this image): " + " ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", *missing], check=True)
    else:
        log("nothing to install: the image has every package")
    import importlib
    importlib.invalidate_caches()
    fetched = {m.split("==", 1)[0].split("[", 1)[0].strip().lower() for m in missing}
    ENV["packages"] = {p: dict(used=installed(p), pinned=PINS[p][1],
                               source="installed here" if p in fetched else "already in the image")
                       for p in REQUIRES}
    for p, v in ENV["packages"].items():
        flag = "" if v["used"] == v["pinned"] else "   (differs from the pin; see section 7)"
        print(f"{p:12s} used {v['used']:14s} pinned {v['pinned']:14s} {v['source']}{flag}")

sys.path[:0] = [str(SRC), str(SRC / "benchmarks"), str(SRC / "examples")]
os.environ.setdefault("MPLBACKEND", "Agg")
