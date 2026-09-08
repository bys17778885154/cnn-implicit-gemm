# conv5

5-layer 3x3 int8 CNN inference on RTX 4090 (SM89) using mma.sync + ldmatrix (no cp.async).

Usage: `conv5.exe [test|bench|gen]`

Build: `powershell build.ps1` (CUDA 12.1 + VS2019 toolset, sm_89)
