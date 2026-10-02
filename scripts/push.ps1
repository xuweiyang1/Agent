<#
.SYNOPSIS
Pushes this repo to GitHub and to the offline server in one step.

.DESCRIPTION
Two remotes are configured, and a single `git push origin main` writes to
both. The server destination is a git URL, not a filesystem path, so this
also works when the server is only reachable through an SSH tunnel.

.PARAMETER GitHubUrl
SSH URL of the GitHub repo, e.g. git@github.com:you/agent.git

.PARAMETER ServerUrl
SSH URL of the server bare repo. In a tunnel setup point the port at the
locally forwarded one, e.g. ssh://you@127.0.0.1:2222/~/agent.git

.EXAMPLE
.\scripts\push.ps1 -GitHubUrl git@github.com:me/agent.git `
                   -ServerUrl ssh://me@127.0.0.1:2222/home/me/agent.git
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$GitHubUrl,
    [Parameter(Mandatory = $true)][string]$ServerUrl
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & git @Args
    if ($LASTEXITCODE -ne 0) { throw "git $($Args -join ' ') failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path (Join-Path $root '.git'))) {
    Write-Host '==> initializing repository' -ForegroundColor Cyan
    Invoke-Git init -b main
}

if (-not (git remote 2>$null | Select-String -Pattern '^origin$' -Quiet)) {
    Invoke-Git remote add origin $GitHubUrl
} else {
    Invoke-Git remote set-url origin $GitHubUrl
}

# Replace any previously configured push destinations so re-running is safe.
$existing = @(git remote get-url --all --push origin 2>$null)
foreach ($url in $existing) { Invoke-Git remote set-url --delete --push origin $url }

Invoke-Git remote set-url --add --push origin $GitHubUrl
Invoke-Git remote set-url --add --push origin $ServerUrl

Write-Host '==> push destinations' -ForegroundColor Cyan
Invoke-Git remote get-url --all --push origin

if (-not (git log --oneline -1 2>$null)) {
    Invoke-Git add -A
    Invoke-Git commit -m 'Initial agent loop: tools, retry, compaction, tests'
}

Write-Host '==> pushing' -ForegroundColor Cyan
Invoke-Git push -u origin main
Write-Host 'done' -ForegroundColor Green