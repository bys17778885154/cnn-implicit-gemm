$ErrorActionPreference = "Continue"
$sdk = "C:\VulkanSDK\1.4.350.0"
$glslc = "$sdk\Bin\glslc.exe"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
New-Item -ItemType Directory -Force -Path build | Out-Null

& $glslc --target-env=vulkan1.3 -O -DSCOPE_SUBGROUP=1 -o build\conv_subgroup.spv shaders\conv5_coopmat.comp
if ($LASTEXITCODE -ne 0) { Write-Output "shader compile failed"; exit 1 }
& $glslc --target-env=vulkan1.3 -O -DSCOPE_SUBGROUP=0 -o build\conv_workgroup.spv shaders\conv5_coopmat.comp
if ($LASTEXITCODE -ne 0) { Write-Output "shader compile failed"; exit 1 }
& $glslc --target-env=vulkan1.3 -O -DSCOPE_SUBGROUP=1 -DUSE_N16 -o build\conv_n16_subgroup.spv shaders\conv5_coopmat.comp
if ($LASTEXITCODE -ne 0) { Write-Output "shader compile failed"; exit 1 }
& $glslc --target-env=vulkan1.3 -O -DSCOPE_SUBGROUP=0 -DUSE_N16 -o build\conv_n16_workgroup.spv shaders\conv5_coopmat.comp
if ($LASTEXITCODE -ne 0) { Write-Output "shader compile failed"; exit 1 }
& $glslc --target-env=vulkan1.3 -O -DSCOPE_SUBGROUP=1 -o build\smoke_subgroup.spv shaders\smoke.comp
if ($LASTEXITCODE -ne 0) { Write-Output "smoke compile failed"; exit 1 }
& $glslc --target-env=vulkan1.3 -O -DSCOPE_SUBGROUP=0 -o build\smoke_workgroup.spv shaders\smoke.comp
if ($LASTEXITCODE -ne 0) { Write-Output "smoke compile failed"; exit 1 }

$src = "src\main.cpp", "..\common\reference.cpp", "..\common\gen_weights.cpp"
cmd /c "call `"$vcvars`" && cl /nologo /O2 /std:c++17 /openmp /I$sdk\Include /I..\common $($src -join ' ') /Fe:build\vkconv5.exe /Fo:build\ /link /LIBPATH:$sdk\Lib vulkan-1.lib"
if ($LASTEXITCODE -eq 0 -and (Test-Path build\vkconv5.exe)) { Write-Output "BUILD OK" } else { Write-Output "BUILD FAILED"; exit 1 }
