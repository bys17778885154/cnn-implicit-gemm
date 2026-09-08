$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
New-Item -ItemType Directory -Force -Path build | Out-Null
$bindir = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
$src = "src\main.cpp","src\reference.cpp","tools\gen_weights.cpp","src\conv_naive.cu","src\conv_tiled.cu","src\mma_test.cu","src\conv_mma.cu"
cmd /c "call `"$vcvars`" && nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler `"/openmp /wd4819`" `"-ccbin=$bindir`" -o build\conv5.exe $($src -join ' ')"
if ($LASTEXITCODE -eq 0 -and (Test-Path build\conv5.exe)) { Write-Output "BUILD OK" } else { Write-Output "BUILD FAILED"; exit 1 }
