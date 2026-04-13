# sweep template

little helper scripts i used to collect my round 2 data. dropping them here
so the rest of the team can copy and tweak them for their own parameter
configs without having to start from scratch.

## what each file does

- **sweep.py** - hits the deployed backend `/evolve/sync` endpoint with each
  config, dumps the json response into `runs/`. edit the `CONFIGS` list and
  the `BASE_REQ` dict at the top to swap in whatever parameter you are
  testing.
- **build_csv.py** - reads `runs/*.json` and flattens them into a single
  csv. one row per (config, strategy, generation).
- **build_plots.py** - reads the csv (well, technically it reads the json
  again, same thing) and makes the 4 matplotlib charts the rubric wants:
  fitness, benchmark compare, steps, runtime.

## how to run it

copy this folder into your own `data/<your_name>/`, then from inside that
folder:

```bash
python sweep.py        # ~12 minutes, hits the live backend
python build_csv.py    # writes <your_name>_data.csv
python build_plots.py  # writes plots/*.png
```

after that you write your own report docx by hand using the csv and the
plots. that part is on you - the doc should be in your own voice.

## things to watch out for

- the live backend can take a couple minutes per full run, so the whole
  sweep is ~12 min. dont kill it early.
- keep `seed` fixed across all your runs in `BASE_REQ` so any difference
  between configs is from the parameter you are testing, not from rng.
- if you change the file name of the csv, update `OUT` near the top of
  `build_csv.py`.
