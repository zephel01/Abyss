"""tools/prebuilt.py - the prebuilt CUDA engine: no CUDA toolkit, Visual Studio or compiler on the user's PC.

GitHub Actions (.github/workflows/prebuilt.yml) compiles the engine for Windows x64 and Linux x64 with CUDA 13.3, for
RTX 30 (sm_86), RTX 40 (sm_89) and RTX 50 (sm_120a) in one binary - the CUDA runtime picks the GPU's own code at
run time, and the engine's per-GPU choices (moe_own's tuned_device, mv_tune, PDL) ask the device, not the build -
and packs it with the CUDA libraries it loads (cuBLAS, cuBLASLt; cudart on Linux) into one archive a platform:

    abyss-engine-cuda-linux-x64.tar.gz      abyss (+ strata, a link to it), abyss_gpu_tune, libcudart / libcublas /
                                            libcublasLt .so
    abyss-engine-cuda-windows-x64.zip       abyss.exe, abyss_gpu_tune.exe, cublas64 / cublasLt64 .dll
    SHA256SUMS                              one "<sha256>  <name>" line per archive

attached to the release of each tag.  The setups (abyss.py through ./abyss.sh / START-ABYSS.bat, and
tools/abyss_setup.py through ./setup-deepseek.sh / setup-deepseek.bat) take it when nvcc is not installed, or with
--prebuilt: the release of this checkout's VERSION (else the latest one), the SHA-256 from SHA256SUMS checked, the
files unpacked into build/ - where the compiled engine would be - and the setup goes on as after a build.

The driver: CUDA 13.3 belongs to the R610 driver branch (Linux 610.43.02 or newer); under CUDA's minor version
compatibility a CUDA 13.x program runs on drivers 580 and newer.  Below 580 the prebuilt engine is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "zephel01/Abyss"
SUMS = "SHA256SUMS"
MANIFEST = "PREBUILT.json"                 # inside the archive: version, commit, archs, CUDA version
ARCHS = (86, 89, 120)                      # the compute capabilities the binary carries code for (sm_120a: 12.0 only)
CUDA = "13.3"
DRIVER_MIN = 580                           # CUDA 13.x under minor version compatibility
DRIVER_RECOMMENDED = 610                   # CUDA 13.3's own branch (R610; Linux 610.43.02)
DRIVER_PAGE = "https://www.nvidia.com/drivers"
ENV_URL = "ABYSS_PREBUILT_URL"             # a folder (URL, file:// or path) holding SHA256SUMS and the archives: tests,
                                           # mirrors, a CI artifact unpacked by hand


def asset_name(system: str | None = None, machine: str | None = None) -> str | None:
    """The archive for this PC, or None when none is built for it (aarch64, macOS)."""
    system = system or ("windows" if os.name == "nt" else sys.platform)
    machine = (machine or platform.machine()).lower()
    if machine not in ("x86_64", "amd64", "x64"):
        return None
    if system.startswith("win"):
        return "abyss-engine-cuda-windows-x64.zip"
    if system.startswith("linux"):
        return "abyss-engine-cuda-linux-x64.tar.gz"
    return None


def version(root: Path = ROOT) -> str:
    try:
        return (root / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# ------------------------------------------------------------------------------------------------ the PC
def driver_version(smi_out: str | None = None) -> str | None:
    """The NVIDIA driver's version ("610.43.02"), the oldest when several GPUs answer; None without nvidia-smi."""
    if smi_out is None:
        try:
            smi_out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    vs = [ln.strip() for ln in (smi_out or "").splitlines() if re.match(r"^\d+(\.\d+)*$", ln.strip())]
    if not vs:
        return None
    return min(vs, key=lambda v: tuple(int(x) for x in v.split(".")))


def driver_verdict(ver: str | None) -> tuple:
    """("ok" | "warn" | "stop", message) for the prebuilt engine on this driver."""
    if not ver:
        return ("stop", "the NVIDIA driver's version is unknown (nvidia-smi did not answer): install the driver "
                        f"{DRIVER_RECOMMENDED} or newer ({DRIVER_MIN} at least) - {DRIVER_PAGE}")
    try:
        major = int(ver.split(".")[0])
    except ValueError:
        major = 0
    if major < DRIVER_MIN:
        return ("stop", f"the NVIDIA driver {ver} is too old for the prebuilt engine (CUDA {CUDA}): it needs "
                        f"{DRIVER_MIN} or newer, {DRIVER_RECOMMENDED} or newer recommended - {DRIVER_PAGE}")
    if major < DRIVER_RECOMMENDED:
        return ("warn", f"the NVIDIA driver {ver} runs the prebuilt engine (CUDA {CUDA}) under CUDA's minor version "
                        f"compatibility (untested on {major}); {DRIVER_RECOMMENDED} or newer is recommended - "
                        f"{DRIVER_PAGE}")
    return ("ok", f"NVIDIA driver {ver} (the prebuilt engine is CUDA {CUDA}: {DRIVER_RECOMMENDED}+ recommended)")


def unsupported(archs) -> list:
    """The compute capabilities (86, 120, ...) the prebuilt binary has no code for."""
    return [int(str(a).rstrip("a")) for a in archs if int(str(a).rstrip("a")) not in ARCHS]


def want_prebuilt(flag: bool, nvcc: str | None, compiler_ok: bool = True) -> bool:
    """--prebuilt, or no toolkit (or, on Windows, no Visual Studio) to compile with: compiling stays the default."""
    return bool(flag) or not nvcc or not compiler_ok


# ------------------------------------------------------------------------------------------------ the release
def parse_sums(text: str) -> dict:
    """{name: sha256} from sha256sum's format ("<hex>  <name>", or "<hex> *<name>")."""
    sums = {}
    for ln in (text or "").splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(\S.*?)\s*$", ln)
        if m:
            sums[Path(m.group(2)).name] = m.group(1).lower()
    return sums


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _get(url: str, timeout=30) -> bytes:
    if not url.startswith(("http://", "https://")):
        return Path(url[7:] if url.startswith("file://") else url).read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "abyss-setup", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def release_bases(ver: str, repo: str = REPO) -> list:
    """Where to look, in order: ABYSS_PREBUILT_URL; this checkout's release (v<VERSION>); the latest release."""
    if os.environ.get(ENV_URL):
        return [os.environ[ENV_URL].rstrip("/")]
    bases = [f"https://github.com/{repo}/releases/download/v{ver}"] if ver else []
    return bases + [f"https://github.com/{repo}/releases/latest/download"]


def find_release(name: str, ver: str, repo: str = REPO, get=_get) -> tuple:
    """(base URL, sha256) of the first release that has `name` in its SHA256SUMS; (None, why) when none has."""
    why = []
    for base in release_bases(ver, repo):
        try:
            sums = parse_sums(get(f"{base}/{SUMS}").decode("utf-8", "replace"))
        except (OSError, ValueError) as e:
            why.append(f"{base}/{SUMS}: {getattr(e, 'reason', None) or e}")
            continue
        if name in sums:
            return base, sums[name]
        why.append(f"{base}/{SUMS} lists no {name}")
    return None, "; ".join(why)


def _safe_members(names) -> None:
    for n in names:
        p = Path(n.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", n):
            raise ValueError(f"unsafe path in the archive: {n}")


def unpack(archive: Path, dest: Path) -> list:
    """The archive's files into dest (its one top folder, if it has one, dropped); their names."""
    tmp = dest / ".prebuilt-unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            _safe_members(z.namelist())
            z.extractall(tmp)
    else:
        with tarfile.open(archive, "r:*") as t:
            members = t.getmembers()
            _safe_members(m.name for m in members)
            for m in members:
                if not (m.isfile() or m.isdir() or m.issym()):
                    raise ValueError(f"unexpected entry in the archive: {m.name}")
                if m.issym() and (Path(m.linkname).is_absolute() or ".." in Path(m.linkname).parts):
                    raise ValueError(f"unsafe link in the archive: {m.name} -> {m.linkname}")
            if hasattr(tarfile, "data_filter"):
                t.extractall(tmp, filter="data")
            else:
                t.extractall(tmp)
    top = [p for p in tmp.iterdir()]
    src = top[0] if len(top) == 1 and top[0].is_dir() else tmp
    names = []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        out = dest / rel
        if p.is_dir() and not p.is_symlink():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_symlink() or out.exists():
            try:
                out.unlink()
            except OSError:                     # (Windows: an engine still running keeps its DLL; a rename frees it)
                out.rename(out.with_name(out.name + f".old{int(time.time())}"))
        if p.is_symlink():
            os.symlink(os.readlink(p), out)
        else:
            shutil.move(str(p), str(out))
        names.append(str(rel))
    shutil.rmtree(tmp, ignore_errors=True)
    return names


def install(dest: Path, ver: str | None = None, repo: str = REPO, log=print, download=None) -> dict:
    """Download, check and unpack the prebuilt engine into dest (build/).  Returns the archive's manifest (version,
    commit, archs, cuda) plus "sha256"; raises RuntimeError with the reason when it cannot."""
    name = asset_name()
    if name is None:
        raise RuntimeError(f"no prebuilt engine for {platform.system()} {platform.machine()} (Windows x64 and Linux "
                           "x64 only): install the CUDA toolkit to compile it")
    ver = version() if ver is None else ver
    base, sha = find_release(name, ver, repo)
    if base is None:
        raise RuntimeError(f"no release with {name} was found ({sha})")
    dl = dest / ".prebuilt-download"
    dl.mkdir(parents=True, exist_ok=True)
    archive = dl / name
    if archive.exists() and sha256_of(archive) != sha:
        archive.unlink()
    for mark in (archive.with_name(archive.name + ".done"),):   # (setup.download's mark: the sum decides here)
        mark.unlink(missing_ok=True)
    if not archive.exists():
        log(f"  downloading {base}/{name}")
        if download is not None:
            download(f"{base}/{name}", archive, name)
        else:
            archive.write_bytes(_get(f"{base}/{name}", timeout=600))
    got = sha256_of(archive)
    if got != sha:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"{name}'s SHA-256 is {got}, {SUMS} says {sha}: the download is damaged (deleted - run "
                           "the setup again)")
    log(f"  SHA-256 checked: {sha}")
    dest.mkdir(parents=True, exist_ok=True)
    unpack(archive, dest)
    try:
        manifest = json.loads((dest / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    exe = dest / ("abyss.exe" if name.endswith(".zip") else "abyss")
    if not exe.exists():
        raise RuntimeError(f"{name} has no {exe.name}")
    # the old name (v1.3.164: strata became abyss) for configs and scripts that still start it: the tar.gz carries a
    # link; on Windows a strata.exe left from a build gets this engine (the zip does not carry a second copy)
    alias = exe.with_name("strata" + exe.suffix)
    try:
        if exe.suffix == ".exe":
            if alias.exists():
                shutil.copy2(exe, alias)
        elif not (alias.is_symlink() and os.readlink(alias) == exe.name):
            alias.unlink(missing_ok=True)       # (an engine compiled before under the old name)
            os.symlink(exe.name, alias)
    except OSError as e:                         # (a strata.exe still running keeps the old one)
        log(f"  (could not update {alias.name}, the engine's old name: {e})")
    if os.name != "nt":
        for p in (exe, dest / "abyss_gpu_tune"):
            if p.exists():
                p.chmod(p.stat().st_mode | 0o755)
    shutil.rmtree(dl, ignore_errors=True)
    return {**manifest, "sha256": sha, "asset": f"{base}/{name}"}


def current(meta: dict) -> bool:
    """The build stamp's prebuilt engine was installed for this checkout's version (the release's own version may be
    older: the latest release, when this version has none yet) - nothing to download again."""
    return bool(meta.get("prebuilt")) and (meta.get("prebuilt_for") or meta.get("prebuilt")) == version()


def stamp_fields(manifest: dict) -> dict:
    """What a setup records in its build stamp for a prebuilt engine ("prebuilt_for": the checkout's version then)."""
    return {"prebuilt": manifest.get("version") or "?", "prebuilt_for": version(),
            "prebuilt_sha256": manifest.get("sha256"),
            "prebuilt_asset": manifest.get("asset"), "cuda": manifest.get("cuda", CUDA),
            "archs": [int(str(a).rstrip("a")) for a in manifest.get("archs") or ARCHS]}
