### VVIDRA - *Vector-Valued Individualized Desirability and Reachability Advisor for Countermeasure Protocol Selection*

##### Proof of concept implementation of the **feasibility-desirability architecture** for personalized decision support in human digital twins

**Status:** Early development (Phase 2 of 12)

##### Overview
'vvidra-poc' is the computational core of VVIDRA, developed as part of PhD thesis "Systems Biology and Personalized Medicine for Optimizing Human Performance in Aging Athletes" (thesis title will most likely change). The motivating application is astronaut health monitoring, but the core is intentionally domain-agnostic: markers, objectives, and interventions are supplied as configuration, not hard-coded. 

The architecture couples two independently specified models:
- **Feasibility model:** a hierarchical, discrete-time stochasstic state-space model that projects the distribution of reachable marker states forward in time under a candidate intervention. It is represented as a weighted particle cloud and personalized through Bayesian posterior updating, starting from cohort-level priors. 
- **Desirability model:** a graded, vector-valued objective field with asymmetric deviation penalties (a modified Derringer-Suich formulation) anchored to a personalized reference point. Acceptability thresholds are annotations on the field, not hard constraints. 

The two models have **independent provenance**: neither is derived from the other. The **coupling** stage evaluates the desirability field over the feasibility cloud and returns, for each candidate intervention, a full distribution of desirability vectors (a *desirability cloud*) together with the probability mass at each threshold. This output is designed to feed a later ranking stage that serves human decision support rather than autonomous optimization. Ranking is outside the scope of this particular PoC.

See included documentation from C:\Users\kevin\proj\vvidra-poc\documentation for full rationale. 

##### Package structure:

```
vvidra-poc/
├── pyproject.toml                          # project metadata, dependencies, tool configuration
├── requirements-lock.txt                   # exact pinned versions used to produce results
├── src/
│   ├── config/                             # system specification: markers, units, interventions,
│   ├── modules/                            
│               ├── desirability/           # penalty shapes and the desirability field
│               ├── feasibility/            # turnover dynamics and forward particle propagator
│               ├── coupling/               # applies the desirability field to the particle cloud
│               └── personalization/        # NumPyro measurement and hierarchical models
│   ├── data/                               # data contracts: particle cloud, desirability cloud

├── tests/                                  # pytest suite (behavioral and architectural properties)
├── notebooks/                              # exploration and figures only, no core logic
├── documentation/                          # background literature 
└── outputs/                                # generated results (git-ignored)
```


##### Requirements

```powershell
# Windows (PowerShell)
py -3.12 -m venv .vvidra-venv
.\.vvidra-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

```bash
# Linux / macOS / WSL2
python3.12 -m venv .vvidra-venv
source .vvidra-venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

To reproduce published results exactly, install the pinned versions instead:

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Usage

> Usage examples will be added once the desirability, feasibility and coupling modules
> are implemented (Phases 4–6).

Scripts and notebooks that run inference should configure JAX before any computation:

```python
import numpyro
numpyro.set_host_device_count(4)  # parallel MCMC chains on CPU
numpyro.enable_x64()              # float64 precision
```

## Development

```bash
pytest             # run the test suite
ruff check .       # lint
ruff format .      # format
```

After adding a dependency to `pyproject.toml`, reinstall and refresh the lock file:

```bash
pip install -e ".[dev]"
pip freeze --exclude-editable > requirements-lock.txt
```

##### Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Define a toy instance (markers, horizons, candidate interventions, target ranges) | <status> |
| 2 | Set up the project | In progress |
| 3 | Define data contracts (system specification, particle cloud, desirability cloud) | Planned |
| 4 | Implement the desirability model | Planned |
| 5 | Implement the feasibility model and forward propagator | Planned |
| 6 | Implement the coupling mechanism | Planned |
| 7 | Build the synthetic cohort generator | Planned |
| 8 | Write the measurement model | Planned |
| 9 | Individualize desirability | Planned |
| 10 | Individualize feasibility | Planned |
| 11 | Verify inference quality (simulation-based calibration, predictive coverage) | Planned |
| 12 | Verify architectural properties | Planned |

##### Contact
**Name:** Kevin Alexander Dominey

**email:** kevin.dominey@spacescience.ro

**Affiliations:** 
- National University of Science and Technology POLITEHNICA Bucharest, Romania
- Institute of Space Science, Magurele Romania