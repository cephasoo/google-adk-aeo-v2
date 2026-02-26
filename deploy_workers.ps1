# deploy_workers.ps1
# Automated deployment script for Sonnet & Prose Backend
# Ensures that the 'shared' module is properly packaged into the Cloud Functions before deployment.

Write-Host "=========================================="
Write-Host "🚀 Sonnet & Prose - Worker Deployment Tool"
Write-Host "=========================================="
Write-Host ""
Write-Host "[1/4] Syncing shared utility modules into worker directories..."

# 1. Copy the 'shared' folder into both worker directories
# This ensures that 'from shared.utils import ...' works natively in the Cloud Build environment
# because 'gcloud functions deploy --source .' only uploads the current directory.
# FIX: Ensure destination is cleared first to prevent nesting via Robocopy or Remove/Copy

Write-Host "Cleaning existing shared directories in workers..."
if (Test-Path ".\worker-story\shared") { Remove-Item ".\worker-story\shared" -Recurse -Force }
if (Test-Path ".\worker-feedback\shared") { Remove-Item ".\worker-feedback\shared" -Recurse -Force }

Write-Host "Copying fresh shared libraries..."
Copy-Item -Path ".\shared" -Destination ".\worker-story" -Recurse -Force
Copy-Item -Path ".\shared" -Destination ".\worker-feedback" -Recurse -Force

Write-Host "✅ Modules synchronized."
Write-Host ""
Write-Host "[2/6] Deploying worker-feedback (Dockerized via Cloud Run)..."

Set-Location .\worker-feedback
try {
    # Ensure LOCATION is set or fallback to us-central1
    $region = $env:LOCATION
    if (-not $region) { $region = "us-central1" }
    
    gcloud run deploy process-feedback-logic --source . --region=$region --clear-base-image
}
finally {
    Set-Location ..
}

Write-Host ""
Write-Host "[3/6] Deploying worker-story (Dockerized via Cloud Run)..."

Set-Location .\worker-story
try {
    $region = $env:LOCATION
    if (-not $region) { $region = "us-central1" }
    
    gcloud run deploy process-story-logic --source . --region=$region --clear-base-image
}
finally {
    Set-Location ..
}

Write-Host ""
Write-Host "[4/6] Deploying worker-tracker (Dockerized via Cloud Run)..."

Set-Location .\worker-tracker
try {
    $region = $env:LOCATION
    if (-not $region) { $region = "us-central1" }
    
    gcloud run deploy worker-tracker --source . --region=$region --clear-base-image --memory 1Gi
}
finally {
    Set-Location ..
}

Write-Host ""
Write-Host "[5/6] Deploying mcp-sensory-server (Dockerized via Cloud Run)..."

Set-Location .\mcp-server
try {
    $region = $env:LOCATION
    if (-not $region) { $region = "us-central1" }
    
    gcloud run deploy mcp-sensory-server --source . --region=$region --clear-base-image
}
finally {
    Set-Location ..
}

Write-Host ""
Write-Host "[6/6] Deploying dispatchers (Dockerized via Cloud Run)..."

Set-Location .\dispatcher-story
try {
    $region = $env:LOCATION
    if (-not $region) { $region = "us-central1" }
    
    Write-Host "Deploying start-story-workflow..."
    gcloud run deploy start-story-workflow --source . --region=$region --clear-base-image
}
finally {
    Set-Location ..
}

Set-Location .\dispatcher-feedback
try {
    $region = $env:LOCATION
    if (-not $region) { $region = "us-central1" }
    
    Write-Host "Deploying handle-feedback-workflow..."
    gcloud run deploy handle-feedback-workflow --source . --region=$region --clear-base-image
}
finally {
    Set-Location ..
}

Write-Host ""
Write-Host "✅ Deployment Pipeline Complete! All 6 services have been Dockerized to completely bypass Serverless gRPC bugs."
