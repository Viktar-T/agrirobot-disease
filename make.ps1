# The Makefile targets for the Windows dev box, where make is not installed.
#   .\make.ps1 env
#   .\make.ps1 test
#   .\make.ps1 cache -BB dinov2_l14_reg -RES 224 -DS ibean
# Keep in step with the Makefile; the Makefile is the canonical list (Linux/WSL).

[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Target = 'help',
    [string]$DS = 'ibean',
    [string]$BB = 'dinov2_l14_reg',
    [string]$RES = '224',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ArgsRest = @()
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Invoke-Step {
    param([string[]]$Command)
    Write-Host "> $($Command -join ' ')" -ForegroundColor DarkCyan
    & $Command[0] @($Command[1..($Command.Length - 1)])
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$py = @('uv', 'run', 'python')

switch ($Target) {
    'help' {
        Write-Host 'targets: env, compute-log, env-check, hub-check, test, lint, fmt, download, manifests, cache, heads, abstain, eval, card, register, bench, probe, serve, clean'
    }
    'env' {
        Invoke-Step @('uv', 'sync')
        if (-not (Test-Path '.env')) {
            Copy-Item '.env.example' '.env'
            Write-Host 'created .env from .env.example - put HF_TOKEN in it' -ForegroundColor Yellow
        }
        Invoke-Step ($py + @('-m', 'ms.compute_log', 'record-env'))
    }
    'compute-log' { Invoke-Step ($py + @('-m', 'ms.compute_log', 'record-env') + $ArgsRest) }
    'env-check' {
        Invoke-Step ($py + @('-c', "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"))
    }
    'hub-check' { Invoke-Step ($py + @('-m', 'ms.backbones.check') + $ArgsRest) }
    'test' { Invoke-Step (@('uv', 'run', 'pytest') + $ArgsRest) }
    'lint' {
        Invoke-Step @('uv', 'run', 'ruff', 'check', '.')
        Invoke-Step @('uv', 'run', 'ruff', 'format', '--check', '.')
    }
    'fmt' {
        Invoke-Step @('uv', 'run', 'ruff', 'format', '.')
        Invoke-Step @('uv', 'run', 'ruff', 'check', '--fix', '.')
    }
    'download'  { Invoke-Step ($py + @('-m', 'ms.data.download', '--dataset', $DS) + $ArgsRest) }
    'manifests' { Invoke-Step ($py + @('-m', 'ms.data.manifests', 'build', '--dataset', $DS) + $ArgsRest) }
    'cache'     { Invoke-Step ($py + @('-m', 'ms.cache.extract', '--backbone', $BB, '--res', $RES, '--dataset', $DS) + $ArgsRest) }
    'heads'     { Invoke-Step ($py + @('-m', 'ms.heads.train', '--backbone', $BB, '--res', $RES) + $ArgsRest) }
    'abstain'   { Invoke-Step ($py + @('-m', 'ms.abstain.fit') + $ArgsRest) }
    'eval'      { Invoke-Step ($py + @('-m', 'ms.eval.run') + $ArgsRest) }
    'card'      { Invoke-Step ($py + @('-m', 'ms.registry', 'card') + $ArgsRest) }
    'register'  {
        Invoke-Step ($py + @('-m', 'ms.registry', 'log') + $ArgsRest)
        Invoke-Step ($py + @('-m', 'ms.registry', 'register') + $ArgsRest)
    }
    'bench'     { Invoke-Step ($py + @('-m', 'ms.service.bench') + $ArgsRest) }
    'probe'     { Invoke-Step ($py + @('-m', 'ms.eval.probe') + $ArgsRest) }
    'serve'     { Invoke-Step @('uv', 'run', 'uvicorn', 'ms.service.app:app', '--host', '127.0.0.1', '--port', '8000') }
    'clean' {
        foreach ($d in @('.pytest_cache', '.ruff_cache')) {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
    }
    default {
        Write-Error "unknown target '$Target' (try: .\make.ps1 help)"
        exit 2
    }
}
