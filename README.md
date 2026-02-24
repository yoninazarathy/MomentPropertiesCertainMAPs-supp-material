# MomentPropertiesCertainMAPs-supp-material

Supplementary material for: Moment Properties for Certain Classes of Structured Markovian Arrival Processes

This notebook supports the paper _Moment Properties for Certain Classes of Structured Markovian Arrival Processes_ by Azam Asanjarani, Sophie Hautphenne, and Yoni Nazarathy.

The moment count calculations are in `notebooks/MTCP_moments.ipynb`. This is a Julia notebook.

The diagonal $C$ matrix benchmarks (SMMPP vs MMTCP timing and relative differences, Section 5 of the paper) are in `experiments/variance_timing_figure.jl`.

The MAP/M/1 queue-length matching experiment (MMTCP warm start vs random start, Section 5 of the paper) is in `experiments/queue_length_matching.jl`.

## Running

Install Julia (≥ 1.9) and activate the project environment:

```bash
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

**Notebook** — open `notebooks/MTCP_moments.ipynb` in Jupyter with the Julia kernel:

```bash
julia -e 'using Pkg; Pkg.activate("."); using IJulia; notebook(dir="notebooks")'
```

**Experiments** — run from the repository root:

```bash
julia experiments/variance_timing_figure.jl
julia experiments/queue_length_matching.jl
```
