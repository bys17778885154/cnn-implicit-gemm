$ErrorActionPreference = "Continue"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
$bindir = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
$cutlass = "D:\b00852572\cutlass-src"
New-Item -ItemType Directory -Force -Path build | Out-Null
cmd /c "call `"$vcvars`" && nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler `"/openmp /wd4819`" `"-ccbin=$bindir`" -I$cutlass\include -I$cutlass\tools\util\include -I..\common -o build\cutlass_conv.exe cutlass_conv.cu ..\common\reference.cpp ..\common\gen_weights.cpp"
if ($LASTEXITCODE -eq 0 -and (Test-Path build\cutlass_conv.exe)) { Write-Output "BUILD OK" } else { Write-Output "BUILD FAILED"; exit 1 }
