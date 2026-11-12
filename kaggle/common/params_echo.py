# Write every parameter above into params.json, together with where the code came from.
json.dump(dict(PARAMS, _source=dict(repo=REPO, commit=COMMIT), _project="{{PROJECT}}"),
          open(WORK / "params.json", "w"), indent=1, default=str)
for k, v in PARAMS.items():
    print(f"{k:24s} {v}")
log("parameters written to params.json")
