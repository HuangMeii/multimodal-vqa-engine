# start-docker.ps1
# Script tự động start Docker Engine trong WSL2 và chạy docker-compose
# Dùng khi đã tắt Docker Desktop GUI

param(
    [switch]$Build,
    [switch]$Down,
    [string]$Action = "up"  # up, down, restart
)

# Màu sắc cho console
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"

function Write-Color($Text, $Color) {
    Write-Host $Text -ForegroundColor $Color
}

# Bước 1: Kiểm tra Docker Engine trong WSL2
Write-Color "`n[1/3] Checking Docker Engine in WSL2..." $Yellow

$dockerdRunning = wsl -d docker-desktop --exec pgrep dockerd 2>$null
if (-not $dockerdRunning) {
    Write-Color "  -> Docker Engine not running. Starting it..." $Yellow
    wsl -d docker-desktop --exec sudo dockerd > $null 2>&1 &
    Start-Sleep -Seconds 3
    Write-Color "  -> Docker Engine started!" $Green
} else {
    Write-Color "  -> Docker Engine is already running." $Green
}

# Bước 2: Kiểm tra Docker context
Write-Color "`n[2/3] Checking Docker context..." $Yellow
$currentContext = docker context inspect --format '{{.Name}}' 2>$null
if ($currentContext -ne "default") {
    Write-Color "  -> Switching to default context..." $Yellow
    docker context use default 2>$null
}
Write-Color "  -> Docker context ready." $Green

# Bước 3: Chạy docker-compose
Write-Color "`n[3/3] Running docker-compose..." $Yellow

$composeDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Down) {
    Write-Color "  -> docker-compose down..." $Yellow
    Push-Location $composeDir
    docker-compose down
    Pop-Location
}
elseif ($Action -eq "restart") {
    Write-Color "  -> docker-compose down && up --build..." $Yellow
    Push-Location $composeDir
    docker-compose down
    docker-compose up --build -d
    Pop-Location
}
else {
    if ($Build) {
        Write-Color "  -> docker-compose up --build..." $Yellow
        Push-Location $composeDir
        docker-compose up --build -d
        Pop-Location
    } else {
        Write-Color "  -> docker-compose up..." $Yellow
        Push-Location $composeDir
        docker-compose up -d
        Pop-Location
    }
}

# Kiểm tra kết quả
Write-Color "`nDone! Checking containers..." $Green
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
