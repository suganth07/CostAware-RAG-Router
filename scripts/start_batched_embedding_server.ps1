param(
    [int]$Port = 11435,
    [int]$Parallelism = 128,
    [int]$ContextSize = 32768,
    [string]$Model = "nomic-embed-text",
    [switch]$RefreshRuntime
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ollamaRoot = Join-Path $env:LOCALAPPDATA "Programs\Ollama\lib\ollama"
$runtime = Join-Path $root "artifacts\cache\llama_server_runtime"
$server = Join-Path $runtime "llama-server.exe"

if ($RefreshRuntime -or -not (Test-Path -LiteralPath $server)) {
    $sourceServer = Join-Path $ollamaRoot "llama-server.exe"
    if (-not (Test-Path -LiteralPath $sourceServer)) {
        throw "Ollama's llama-server.exe was not found at $sourceServer. Install/update Ollama first."
    }
    New-Item -ItemType Directory -Force -Path $runtime | Out-Null
    Copy-Item -Path (Join-Path $ollamaRoot "*.dll") -Destination $runtime -Force
    Copy-Item -LiteralPath $sourceServer -Destination $server -Force

    $cudaRuntime = Get-ChildItem -LiteralPath $ollamaRoot -Directory |
        Where-Object { $_.Name -like "cuda_v*" } |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if ($null -eq $cudaRuntime) {
        throw "No CUDA runtime was found under $ollamaRoot."
    }
    Copy-Item -Path (Join-Path $cudaRuntime.FullName "*.dll") -Destination $runtime -Force
}

$manifestPath = Join-Path $HOME ".ollama\models\manifests\registry.ollama.ai\library\$Model\latest"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "The Ollama model manifest was not found. Run: ollama pull $Model"
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$modelLayer = $manifest.layers |
    Where-Object { $_.mediaType -eq "application/vnd.ollama.image.model" } |
    Select-Object -First 1
if ($null -eq $modelLayer) {
    throw "The model layer is missing from $manifestPath."
}
$blobName = [string]$modelLayer.digest -replace ":", "-"
$modelBlob = Join-Path $HOME ".ollama\models\blobs\$blobName"
if (-not (Test-Path -LiteralPath $modelBlob)) {
    throw "The model blob was not found at $modelBlob."
}

Write-Host "Starting batched embeddings on http://127.0.0.1:$Port"
Write-Host "Parallelism=$Parallelism ContextSize=$ContextSize (foreground; press Ctrl+C to stop)"
& $server @(
    "--model", $modelBlob,
    "--port", $Port,
    "--host", "127.0.0.1",
    "--no-webui",
    "--offline",
    "--embedding",
    "--pooling", "mean",
    "-b", "32768",
    "-ub", "32768",
    "-np", $Parallelism,
    "-c", $ContextSize,
    "-ngl", "999",
    "--no-mmap",
    "--no-cache-prompt",
    "--log-disable"
)
