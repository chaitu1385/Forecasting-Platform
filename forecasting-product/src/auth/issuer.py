"""Write operator credentials to new, owner-only files without console output."""

import os
import stat
import subprocess
from pathlib import Path

_WINDOWS_WRITE = r"""
$ErrorActionPreference = 'Stop'
$stream = $null
try {
    $path = $env:FORECAST_PRIVATE_TOKEN_OUTPUT
    $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    $security = [System.Security.AccessControl.FileSecurity]::new()
    $security.SetOwner($sid)
    $security.SetAccessRuleProtection($true, $false)
    $security.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new(
        $sid, [System.Security.AccessControl.FileSystemRights]::FullControl,
        [System.Security.AccessControl.AccessControlType]::Allow))
    $stream = [System.IO.FileStream]::new(
        $path, [System.IO.FileMode]::CreateNew,
        [System.Security.AccessControl.FileSystemRights]::FullControl,
        [System.IO.FileShare]::None, 4096, [System.IO.FileOptions]::None, $security)
    $actual = $stream.GetAccessControl()
    $rules = @($actual.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]))
    if (-not $actual.AreAccessRulesProtected -or $rules.Count -ne 1 -or
        $actual.GetOwner([System.Security.Principal.SecurityIdentifier]).Value -ne $sid.Value -or
        $rules[0].IdentityReference.Value -ne $sid.Value -or $rules[0].IsInherited -or
        $rules[0].AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow -or
        $rules[0].FileSystemRights -ne [System.Security.AccessControl.FileSystemRights]::FullControl) {
        throw 'Owner-only file ACL was not established'
    }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes([Console]::In.ReadToEnd())
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Flush($true)
} catch {
    [Console]::Error.WriteLine('Secure token output failed; choose a new file in an existing ACL-capable directory.')
    exit 1
} finally {
    if ($null -ne $stream) { $stream.Dispose() }
}
"""


def _write_windows(path: Path, token: str) -> None:
    powershell = (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if not powershell.is_file():
        raise RuntimeError("Secure Windows token output requires Windows PowerShell; no insecure fallback is available")
    environment = dict(os.environ)
    environment.pop("JWT_SECRET_KEY", None)
    environment["FORECAST_PRIVATE_TOKEN_OUTPUT"] = str(path)
    try:
        result = subprocess.run(
            [str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_WRITE],
            input=token + "\n", encoding="utf-8", capture_output=True,
            env=environment, timeout=15, check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Secure Windows token output timed out; inspect the requested private file before retrying") from None
    if result.returncode:
        raise RuntimeError(
            "Secure token output failed; use a new file in an existing ACL-capable directory. "
            "An empty or private partial file may remain."
        )


def write_private_token(output_path: Path, token: str) -> None:
    """Create one new private file, refusing overwrite or insecure output.

    POSIX creation uses mode 0600. Windows creation supplies a protected,
    current-user-only DACL atomically through the FileStream security overload,
    verifies it before reading credential input, then writes through the handle.
    No credential is passed in a command argument or emitted to a console.
    """
    if not isinstance(token, str) or not token or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("Token output must be a nonempty single-line encoded credential")
    if str(output_path) in {"", "-", "."}:
        raise ValueError("Provide an explicit new private file, not a directory or standard output")
    path = Path(output_path).expanduser().absolute()
    if not path.parent.is_dir():
        raise ValueError("The token output parent directory must already exist")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Token output already exists; choose a new file")
    if os.name == "nt":
        if ":" in path.name or str(path).startswith(("\\\\.\\", "\\\\?\\")) or path.is_reserved():
            raise ValueError("Token output must be a regular file, not a device or alternate data stream")
        _write_windows(path, token)
        return
    if os.name != "posix":
        raise RuntimeError("Owner-only token output is not supported on this operating system")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
            raise RuntimeError("Owner-only token output could not be established; no credential was written")
        with os.fdopen(descriptor, "w", encoding="ascii") as output:
            descriptor = -1
            output.write(token + "\n")
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor != -1:
            os.close(descriptor)
