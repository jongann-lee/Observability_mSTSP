# Navigating Uncertain Environments with Heterogeneous Visibility

Codebase for the paper 'Navigating Uncertain Environments with Heterogeneous Visibility'.

Two agents are compared:
- **Shortest Path (SP) Agent** — plans via shortest paths and replans when a blockage is discovered.
- **RPP Agent** - plans using the reactive path problem method by Macdonald et al.
- **Our Agent** — uses a visibility-aware reward function to proactively route through informative edges, reducing expected travel cost.

## Requirements

- Python >= 3.12
- See `requirements.txt` for Python package dependencies.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
.
├── Graph_Generation/       # Graph construction, visibility, edge blocking, target graph
├── Single_Agent/           # Agent algorithms (RepeatedTopK, LKH TSP, RPP, reward functions)
├── Multi_Agent_TSP/        # Multi-agent TSP solver (experimental)
├── Automatic_Generated_Maps/  # Procedurally generated plateau environments (Section V-B)
├── Real_Life_Maps/         # Real-world natural terrain environment (Section V-C)
├── benchmark.py            # Single plateau benchmark (Section V-A, V-D)
├── single_plateau.ipynb           # Visualization for plateau benchmark
├── requirements.txt
└── README.md
```

## Experiments

### 1. Plateau Environment

A hand-crafted grid map with four plateaus and manually defined chokepoints. Runs the selected agent over many random blockage realizations and reports mean path cost. Two pre-built maps are provided:

- `plateau_16.pkl` — 16x16 grid, used for Section V-A results
- `plateau_12.pkl` — 12x12 grid, used for Section V-D results


**Run the benchmark:**

```bash
python benchmark.py
```

Configure which agent to run by editing the flags at the top of `benchmark.py`:

```python
use_shortest_path_agent = False
use_our_agent = True
use_RPP_agent = False
```

Other settings (`num_runs`, `edge_block_prob`, target graph parameters) can also be edited directly in the file.

**Visualize results:**

Open `single_plateau.ipynb` in Jupyter to visualize the plateau map, agent trajectories, and edge usage.

```bash
jupyter notebook single_plateau.ipynb
```

### 2. Procedurally Generated Environment

Generates a suite of maps with procedural blobs (plateaus) and chokepoints, then benchmarks both agents on each map.

**Run the benchmark:**

```bash
cd Automatic_Generated_Maps
python proc_gen_benchmark.py [OPTIONS]
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--num-maps` | 50 | Number of maps to generate and benchmark |
| `--num-runs` | 200 | Blockage realizations per map |
| `--seed-start` | 3000 | Starting random seed for map generation |
| `--output` | `benchmark_results.csv` | Per-map results CSV |
| `--output-summary` | `benchmark_summary.json` | Aggregate summary JSON |
| `--output-maps` | `top_bottom_maps.pkl` | Top 3 / bottom 3 maps by improvement |

**Visualize maps:**

Open `Automatic_Generated_Maps/visualize_maps.ipynb` to inspect individual generated maps, chokepoint placement, and agent path comparisons.

Open `Automatic_Generated_Maps/benchmark_histograms.ipynb` to view histograms and summary statistics from benchmark results.

### 3. Natural Terrain Environment

Uses a real Digital Elevation Model (DEM) GeoTIFF to build a terrain graph where edge costs reflect elevation changes. Obstacle ovals are placed along the shortest path to create chokepoints.

**Run the benchmark:**

```bash
python Real_Life_Maps/real_map_benchmark.py [OPTIONS]
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--dem-path` | `Real_Life_Maps/WV_DEM.tif` | Path to the DEM GeoTIFF file |
| `--grid-size` | 64 | Grid resampling size (NxN) |
| `--num-runs` | 200 | Blockage realizations |
| `--block-prob` | 0.5 | Per-oval blocking probability |
| `--reward-ratio` | 1.0 | RepeatedTopK reward ratio |
| `--num-sp-ovals` | 3 | Number of obstacle ovals on shortest path |
| `--num-diverse-ovals` | 0 | Number of obstacle ovals on diverse paths |
| `--output` | `real_map_results.csv` | Per-run results CSV |
| `--output-summary` | `real_map_summary.json` | Summary JSON |

There is also an alternative script `Real_Life_Maps/benchmark_realmap.py` that uses a simpler configuration-toggle approach (edit flags at the top of the file) with fixed obstacle placements.

**Visualize results:**

Open `Real_Life_Maps/visualization.ipynb` to view the terrain, obstacle placements, and agent paths.

Note: The real map experiments require `rasterio` for loading GeoTIFF files. The DEM file `Real_Life_Maps/WV_DEM.tif` is included in the repository.
