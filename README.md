# LLM Serving Simulation – Final Project

## Project overview

This project develops a stochastic discrete‑event simulation of an LLM (large language model) inference serving system, focusing on how scheduling and batching policies affect throughput and latency experienced by users. The model represents user queries as jobs that pass through a compute‑intensive prefill phase followed by a lighter decode phase, and tracks key metrics including throughput, Time to First Token (TTFT), Time Between Tokens (TBT), tail latency, and GPU utilization. The simulation is used to compare a baseline run‑to‑completion scheduler with more advanced prefill‑prioritizing batching and multi‑GPU configurations, showing how system‑level decisions affect both efficiency and user‑visible performance.

The core simulation logic lives in `simulation.py`, which defines the job/request data structures, the stochastic GPU service‑time model, single‑GPU and multi‑GPU schedulers, and experiment drivers (validation, batch‑size sweeps, scaling experiments, and sensitivity analyses). The final report notebook, `main_report.ipynb`, combines a written report with executable code that calls these functions to generate all figures and quantitative results. 

## Notebook structure (`main_report.ipynb`)

The `main_report.ipynb` notebook is organized into two major parts:

- **Written report (~first 12 pages).**  
  The top portion of the notebook contains the full narrative: motivation, system overview, modeling assumptions, experiment design, results, and discussion. This section reads like a traditional paper and can be viewed directly on GitHub or in Jupyter without executing any code.
- **Appendix (remaining cells).**  
  After the written portion, the notebook includes an appendix section that programmatically calls into `simulation.py` to:
  - Configure the service‑time model and workload parameters.  
  - Run single‑GPU and multi‑GPU simulations under different schedulers.  
  - Validate against queueing‑theory baselines (e.g., M/M/1).  
  - Generate the plots and tables referenced in the main text.  
  This section is intended for reproducibility and for readers who want a detailed view of the implementation and experiments.

## Repository layout

A minimal layout for the repository is:

```
.
├── simulation.py       # Simulation engine (data structures, schedulers, experiments)
├── main_report.ipynb   # Final report + methods/figures appendix
├── data/               # Folder containing all generated CSV data files
└── README.md           # Project description and run instructions
```

All CSV data output by the appendix experiments will be saved in the `data/` folder. This keeps the project directory clean and makes it easier to locate all results files.

`main_report.ipynb` expects `simulation.py` to be in the same directory so that it can be imported directly with `import simulation` without any manual directory changes.

## How to run the notebook

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Install required Python packages**

   The simulation uses standard scientific Python libraries, including `numpy`, `scipy`, `matplotlib`, and `pandas`.

   ```bash
   pip install numpy scipy matplotlib pandas jupyter
   ```

3. **Start Jupyter and open the report**

   ```bash
   jupyter notebook
   ```

   Then, in the Jupyter file browser, open `main_report.ipynb`.

4. **Run the report and appendix**

   - To *read* the report, scroll through the top portion of `main_report.ipynb` (the written report). No execution is required for this part.
   - To *reproduce experiments and plots* in the appendix:
     1. Ensure the first code cell imports the simulator (`import simulation`).
     2. In the Jupyter menu, select **Kernel → Restart & Run All**.
     3. Wait for all cells to finish running; figures and printed metrics will appear in the appendix cells.

## Accessing Generated Data

When you run the experiments in the appendix of `main_report.ipynb`, the simulation results are automatically saved as CSV files in the `data/` directory within your project. These files include raw metrics for the experiments performed.

The generated CSV files are:

- `data_scheduler_a_baseline.csv`: Baseline Scheduler A metrics.
- `data_scheduler_a_throughput_sweep.csv`: Scheduler A throughput vs. arrival rate.
- `data_validation_light_traffic.csv`: Scheduler B light traffic validation.
- `data_capacity_vs_k.csv`: Capacity vs. Batch Size (K).
- `data_scheduler_b_throughput_sweep.csv`: Scheduler B throughput vs. arrival rate.
- `data_batching_scale.csv`: Batching scale experiment results.
- `data_sensitivity_ca.csv`: Sensitivity to cost parameters.
- `data_sensitivity_regime.csv`: Sensitivity to job regimes (short vs long).
- `data_scheduler_comparison.csv`: Comparison between Scheduler A and B.
- `data_multigpu_scaling.csv`: Multi-GPU scaling experiment results.

You can access these files directly in your OS file browser or load them using pandas for further analysis:

```python
import pandas as pd
df = pd.read_csv('data/data_multigpu_scaling.csv')
print(df.head())
```
