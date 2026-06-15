# 3D Phase-Field Simulations : 512^3 via GPU on a Laptop

# BCP-BHM-Blackwell-2026
First benchmark of 512³ 3D phase-field simulations  on consumer NVIDIA Blackwell GPUs via the BHM scheme  in MATLAB 2026a.  


# Why BHM?
Unlike SAV or Convex-Splitting methods, the Biharmonic-Modified (BHM) scheme avoids system enlargement and auxiliary variables. This linear, single-solve structure is essential for memory-dominated regimes, enabling 
double-precision simulations on 24GB GPUs where other methods would typically trigger VRAM overflow.

Companion code for:
Orizaga et al. (2026), *Computer Physics Communications* (under review).

## Requirements
- MATLAB 2026a with Parallel Computing Toolbox
- NVIDIA GPU (RTX 5090 recommended)
- 24GB VRAM for N=512³ simulations

## Contents
- `CH3D_GPU_project_2026_BCP.m`: Main BCP simulation (good up to $N=256$)
-  `CH3D_GPU_project_2026_Nested.m`: Main BCP simulation (nested $N=512$)
- `visualization.m` : Isosurface and cross-section plots

## Usage
```matlab
% Run simulation
CH3D_GPU_project_2026_BCP(dt, M1, iter, tfinal, N)

% Example parameters
dt=0.01; M1=7.5; tfinal=100; N=256;
```
## Output
Running the codes: **CH3D_GPU_project_2026_BCP.m** followed by **visualization.m** <br>
will generate the following plot ($\alpha$ =35 and .39 volume fraction)

<img src="512 tf10.png" width=400px height=400px> <img src="512 tf10_2.png" width=400px height=400px>

## Output
Running the codes: **CH3D_GPU_project_2026_Nested.m** followed by **visualization.m** <br>
will generate the following plot ($\alpha$ = 20 and .25 volume fraction)

<img src="512 tf10a.png" width=400px height=400px> <img src="512 tf10b.png" width=400px height=400px>


## Citation

If you use this code, please cite:

Orizaga,  S. Gonzalez-Parra, G., Fabien, M.,  Owusu, S. (2026).
“A Semi-Implicit Energy-Stable Biharmonic-Modified Scheme for Large-Scale 3D Block Copolymer Simulation on Consumer GPUs”
Manuscript submitted to Computer Physics Communications (CPC).

Code available at:
https://github.com/sauloorizaga/BCP-BHM-Blackwell-2026

### Historical Note:
The BHM methodology was originally developed to resolve the numerical challenges of variable-mobility Cahn-Hilliard equations. However, our research [[Orizaga & Witelski, 2024]](https://www.sciencedirect.com/science/article/pii/S0927025624003665) demonstrated that the scheme provides superior memory efficiency even for constant-mobility systems. This work extends that benefit to the 3D Block Copolymer (BCP) model, leveraging the scheme's minimal memory footprint to break the 512^3 barrier on consumer GPUs.
 
## Contact
We welcome questions, feedback, and potential collaboration opportunities—feel free to reach out! <br>

**Saulo Orizaga** — saulo.orizaga@nmt.edu  <br>
Associate Professor of Mathematics          <br>
New Mexico Institute of Mining and Technology  <br>
Socorro, NM 87801, USA.
