# Builds the DLSS5 worker (exe + caller shim) from the DLSS5-for-Nuke project and
# drops the binaries into dlss5_nodes/runtime/.
#
# Requirements: git, VS2022 Build Tools (C++), CMake, Ninja on PATH.
# Does NOT download or install any NVIDIA binaries.

$ErrorActionPreference = "Stop"
$libRoot = $PSScriptRoot
$srcDir  = Join-Path $libRoot "_worker_src"
$runtime = Join-Path $libRoot "dlss5_nodes\runtime"
$repoUrl = "https://github.com/KJzzzKJ/DLSS5-for-Nuke.git"

if (-not (Test-Path $srcDir)) {
    Write-Host "[1/4] Cloning $repoUrl" -ForegroundColor Yellow
    git clone --depth 1 $repoUrl $srcDir | Out-Null
} else {
    Write-Host "[1/4] Using existing source in $srcDir" -ForegroundColor Yellow
}

# Apply our patches (live mode with per-frame settings, --nr-dir/--sr-dir, SIMD
# pixel conversion, init failure reasons on stderr). Every file in worker_patches
# replaces its namesake in the upstream worker/ folder.
Write-Host "[2/4] Applying worker_patches" -ForegroundColor Yellow
Get-ChildItem (Join-Path $libRoot "worker_patches") -File | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $srcDir "worker\$($_.Name)") -Force
}

$vcvars = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) { throw "vcvars64.bat not found - install VS2022 Build Tools with 'Desktop development with C++'." }

$ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $ninja) { throw "ninja not found on PATH (pip install ninja, or install via VS)." }

$worker = Join-Path $srcDir "worker"
$build  = Join-Path $worker "build"
if (Test-Path $build) { Remove-Item -Recurse -Force $build }
New-Item -ItemType Directory $build | Out-Null

Write-Host "[3/4] Building worker (Release, Ninja)" -ForegroundColor Yellow
$ninjaFwd = $ninja -replace '\\', '/'
cmd.exe /c "call `"$vcvars`" >nul && cd /d `"$build`" && cmake `"$worker`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninjaFwd`" -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release"
if ($LASTEXITCODE -ne 0) { throw "Worker build failed." }

Write-Host "[4/4] Deploying to $runtime" -ForegroundColor Yellow
New-Item -ItemType Directory -Force $runtime | Out-Null
Copy-Item (Join-Path $build "DLSS_Nuke_Worker.exe") (Join-Path $runtime "DLSS5Worker.exe") -Force
Copy-Item (Join-Path $build "nvngx.dll")            (Join-Path $runtime "nvngx.dll") -Force

Write-Host ""
Write-Host "Built:" -ForegroundColor Green
Get-ChildItem $runtime | ForEach-Object { "  {0}  ({1} bytes)" -f $_.Name, $_.Length }
if (-not (Test-Path (Join-Path $runtime "nvngx_dlssnr.dll"))) {
    Write-Host ""
    Write-Host "NOTE: nvngx_dlssnr.dll (NVIDIA DLSS 5 Neural Rendering runtime) is not present." -ForegroundColor Yellow
    Write-Host "      Copy a legitimately obtained copy into $runtime before using the node." -ForegroundColor Yellow
}
