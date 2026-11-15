$ErrorActionPreference = 'Stop'
$deviceId = 'USB\VID_8086&PID_0B07\6&1288C7B4&0&4'
$logPath = Join-Path $PSScriptRoot 'depth_diagnostics\d435-reenumerate.log'
try {
    $device = Get-PnpDevice -InstanceId $deviceId -PresentOnly
    if ($device.InstanceId -notmatch '^USB\\VID_8086&PID_0B07\\') { throw 'Unexpected device' }
    & pnputil /remove-device $deviceId /subtree 2>&1 | Out-File -LiteralPath $logPath -Encoding utf8
    & pnputil /scan-devices 2>&1 | Out-File -LiteralPath $logPath -Encoding utf8 -Append
} catch {
    $_ | Out-File -LiteralPath $logPath -Encoding utf8 -Append
    exit 1
}
