function Show-FleetSummary {
    <#
    .SYNOPSIS
    Shows the VM campaign's selected-policy loss table without changing the run.
    .EXAMPLE
    Show-FleetSummary
    .EXAMPLE
    Show-FleetSummary -Deliveries
    .NOTES
    Uses Windows OpenSSH. It prompts for your SSH password unless key-based
    authentication is configured. No password is stored in this script.
    #>
    [CmdletBinding()]
    param(
        [ValidatePattern('^[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+$')]
        [string]$Vm = 'gal.pinto@132.68.161.40',
        [ValidatePattern('^/[A-Za-z0-9_./-]+$')]
        [string]$Campaign = '/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/runs/icaps_new_cities_v7_all_policies',
        [switch]$Deliveries
    )
    $sshCommand = Get-Command ssh -CommandType Application -ErrorAction Stop
    $pythonPath = '/data/workspace/robot_delivery/dynamic-delivery-robot-fleet/.venv/bin/python'
    $summaryPath = '/data/workspace/robot_delivery/dynamic-delivery-robot-fleet-new-city-benchmarks/operations/show_city_campaign_summary.py'
    $remoteCommand = "'$pythonPath' '$summaryPath' --campaign '$Campaign'"
    if ($Deliveries) { $remoteCommand += ' --deliveries' }
    & $sshCommand.Source -o ConnectTimeout=15 $Vm $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw 'VM summary failed. Check VPN/network access, SSH authentication, and the campaign path.'
    }
}
