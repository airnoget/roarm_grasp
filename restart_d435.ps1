$ErrorActionPreference = 'Stop'
$deviceId = 'USB\VID_8086&PID_0B07\6&1288C7B4&0&4'
$logPath = Join-Path $PSScriptRoot 'depth_diagnostics\d435-restart.log'
try {
    $device = Get-PnpDevice -InstanceId $deviceId
    if ($device.InstanceId -notmatch '^USB\\VID_8086&PID_0B07\\') { throw 'Unexpected device identity' }
    $result = & pnputil /restart-device $deviceId 2>&1
    $result | Out-File -LiteralPath $logPath -Encoding utf8
    $result
} catch {
    $_ | Out-File -LiteralPath $logPath -Encoding utf8
    exit 1
}
