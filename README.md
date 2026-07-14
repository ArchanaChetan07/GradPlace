# GradPlace — Differentiable VLSI Cell Placement

### PyTorch analytical placer with macro-aware density, bin capacity caps, and two-phase λ_overlap ramp for overlap/wirelength optimization.

[![GitHub](https://img.shields.io/badge/repo-GradPlace-181717?logo=github)](https://github.com/ArchanaChetan07/GradPlace)
[![Language](https://img.shields.io/badge/language-Python-3572A5)](https://github.com/ArchanaChetan07/GradPlace)
[![License](https://img.shields.io/badge/license-See%20repository-yellow)](https://github.com/ArchanaChetan07/GradPlace)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/ArchanaChetan07/GradPlace/actions)

---

## Overview

Global placement must spread macros/stdcells while minimizing wirelength and residual overlaps using GPU-friendly gradients.

Large placement.py implements density/overflow potentials with macro_weight=5.0, stdcell bin cap 0.25×bin_capacity, and λ_overlap smooth ramp 0.1→1.0 then →15.0; test harnesses and timestamped output metrics/images.

Runnable placement challenge entrypoint printing average overlap, wirelength, and runtime across predefined test cases.

This repository is maintained as **production-minded portfolio work**: clear architecture, automated checks where present, and metrics that are **traceable to committed artifacts** (never invented).

---

## Architecture

Netlist/test case → initialize cell coords → optimize wirelength+overlap losses with λ schedule → metrics/images under outputs/

```mermaid
flowchart TD
  T[Test case macros/stdcells] --> P[placement optimizer]
  P --> D[density / overflow potential]
  P --> W[wirelength loss]
  D --> L[λ_overlap ramp 0.1→15]
  W --> L
  L --> O[outputs/metrics + images]
```

```mermaid
sequenceDiagram
  participant U as User/Client
  participant S as Service/Pipeline
  participant E as Eval/Tools
  U->>S: request / job
  S->>E: execute
  E-->>S: results
  S-->>U: report / response
```

---

## Results & repository facts

> Only values found in code, configs, tests, or generated reports are listed. Absence of a clinical/ML accuracy number means it was **not** published in-repo.

| Metric | Value | Source |
|---|---|---|
| macro_weight | **5.0** | `placement.py` |
| Standard-cell bin capacity cap | **0.25 × bin_capacity** | `placement.py` |
| λ_overlap ramp range | **0.1 → 15.0 (two-phase)** | `placement.py` |
| Tracked blobs on main | **15** | `git tree main` |
| Tracked files | **15** | `git tree` |
| Python modules | **5** | `git tree` |
| Test-related paths | **1** | `git tree` |
| CI workflows | **Yes** | `.github/workflows` |
| Docker present | **No** | `repo root` |

```mermaid
%%{init: {'theme':'base'}}%%
pie showData title Language composition (bytes)
    "Python" : 100
```

---

## Key features

- Macro-weighted density potential
- Standard-cell bin capacity capping
- Two-phase λ_overlap force ramp
- Debug image dumps (snapshots, heatmaps, loss curves)
- Quick vs full test suites
- CUDA optional

---

## Tech stack

| Layer | Technology |
|---|---|
| language | Python |
| dl | PyTorch |
| domain | VLSI placement |
| tooling | debug visualizations / JSON metrics |

---

## Skills demonstrated

Python · PyTorch · CI/CD · testing · automation

Keyword surface: **Python · Python · machine-learning · CI/CD · testing · API · Docker · automation · data-science · software-engineering · system-design · observability · LLM · cloud**

---

## Project structure

```text
GradPlace/
├── placement.py
├── main.py / src/main.py
├── test.py
├── check_cuda.py
├── outputs/README.md
└── .github/workflows/ci.yml
```

---

## Installation & usage

```bash
git clone https://github.com/ArchanaChetan07/GradPlace.git
cd GradPlace
pip install -r requirements.txt
python main.py --quick --device cpu
```

---

## How it works

Cells are optimized with differentiable density/overflow and wirelength objectives; macros are upweighted in density, stdcells are capacity-capped per bin, and overlap penalty strength ramps across training phases before emitting metrics.

---

## Future improvements

- Commit example outputs/run_* metrics JSONs
- Expand README beyond template spam using outputs/README content
- Public leaderboard numbers if contest scores exist

---

## License

See repository.

---

<p align="center">
  <b>GradPlace — Differentiable VLSI Cell Placement</b><br/>
  <a href="https://github.com/ArchanaChetan07/GradPlace">github.com/ArchanaChetan07/GradPlace</a>
</p>
