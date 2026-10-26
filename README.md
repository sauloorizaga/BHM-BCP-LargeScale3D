# 3D Phase-Field Simulations : 512^3 via GPU on a Laptop

# BHM-BCP-LargeScale3D
Companion code for an efficient computational framework for large-scale 3D block copolymer (BCP) simulations with the Biharmonic-Modified (BHM) scheme in MATLAB. The repository includes the production solver, the convergence studies in time and space, a controlled comparison with convex splitting (CS) and scalar auxiliary variable (SAV) methods, CPU and GPU benchmarks, a single- versus double-precision study, and long-time simulations at N = 512³ in double precision on a single laptop GPU.

# Why BHM?
BHM is a linear, semi-implicit scheme that requires neither a convex–concave splitting of the energy nor an auxiliary variable. With a Fourier pseudo-spectral discretization, each time step costs one forward and one inverse FFT, and the free energy is evaluated through Parseval's identity at no additional FFT cost. In our comparison, BHM attains the accuracy of SAV at the computational cost of CS, and the nested implementation runs a 512³ double-precision simulation with a peak GPU memory of about 11.2 GiB.
Companion code for:
Orizaga et al. (2026), *Engineering With Computers - Springer* (under review).

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
