# Windows PowerShell — run AS ADMINISTRATOR.
#
# Forwards Windows host port 9876 to the WSL2 viewer container so any
# device on the same LAN (phone, tablet, second laptop) can reach it.
#
# Usage:
#   PS> .\expose-on-lan.ps1           # add the forwarder + firewall rule
#   PS> .\expose-on-lan.ps1 -Remove   # tear them back down

param(
  [int]    $Port    = 9876,
  [string] $WslIp   = "",
  [switch] $Remove
)

if ($Remove) {
  Write-Host "Removing portproxy + firewall rule for port $Port..."
  netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 | Out-Null
  Remove-NetFirewallRule -DisplayName "AHP Viewer $Port" -ErrorAction SilentlyContinue
  Write-Host "Done."
  exit 0
}

if ([string]::IsNullOrWhiteSpace($WslIp)) {
  $WslIp = (wsl hostname -I).Trim().Split(" ")[0]
}

Write-Host "Forwarding Windows :$Port -> WSL ${WslIp}:$Port"
netsh interface portproxy add v4tov4 `
  listenport=$Port listenaddress=0.0.0.0 `
  connectport=$Port connectaddress=$WslIp | Out-Null

New-NetFirewallRule -DisplayName "AHP Viewer $Port" `
  -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow `
  -ErrorAction SilentlyContinue | Out-Null

$lanIp = (Get-NetIPAddress -AddressFamily IPv4 |
          Where-Object { $_.PrefixOrigin -eq 'Dhcp' } |
          Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "Open this URL from your phone (same Wi-Fi):"
Write-Host "  http://${lanIp}:$Port" -ForegroundColor Green
Write-Host ""
Write-Host "To tear down later: .\expose-on-lan.ps1 -Remove"
