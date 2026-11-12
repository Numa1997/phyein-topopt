ENV["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
TIMES["total"] = round(time.time() - T0, 3)
json.dump(ENV, open(WORK / "env.json", "w"), indent=1, default=str)
json.dump(dict(machine=dict((k, ENV[k]) for k in ("cpu_model", "cpus_usable", "ram_total_gb", "gpu", "on_kaggle")),
               precision=PRECISION, stages_s=TIMES, wall_s=TIMES["total"]),
          open(WORK / "timing.json", "w"), indent=1)
log(f"timing.json written, total {TIMES['total']:.1f} s")

files = sorted(p for p in WORK.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
with open(WORK / "SHA256SUMS.txt", "w") as f:
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        f.write(f"{h}  {p.relative_to(WORK)}\n")
        print(f"{h[:16]}…  {p.stat().st_size:>10,d} B  {p.relative_to(WORK)}")

failed = [k for k, v in CHECKS.items() if not v["pass"]]
print(f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} verification checks passed.")
if failed:
    raise RuntimeError("verification failed: " + ", ".join(failed))
