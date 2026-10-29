from pathlib import Path
import re
import shutil

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

ROOT = Path("results/benchmarks/ipc/observables")
DEST_ROOT = ROOT.parent / "observables_copy"

# Set to True to preview the copies without creating files.
DRY_RUN = False

# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------


def extract_backend(filename, content):
    """Identify the backend. Files without a backend prefix are shots."""
    name = filename.lower()

    for backend in ("estimator-exact", "spinpulse", "shots"):
        if name.startswith(backend + "_") or name.startswith(backend + "__"):
            return backend

    patterns = [
        r"--backend\s*[= ]\s*(estimator-exact|shots|spinpulse)\b",
        r"\bbackend\s*[:=]\s*(estimator-exact|shots|spinpulse)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    return "shots"


def extract_seed(filename, content):
    """Extract the seed from the filename or log contents."""
    patterns = [
        r"(?:^|[_\-\s])seed[_=-]?(\d+)",
        r"\bseed\s*[:=]\s*(\d+)",
        r"\brandom_seed\s*[:=]\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)

    for pattern in patterns[1:]:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def extract_precision(filename, content):
    """Extract precision; preserve multiple precisions if present."""
    name = filename.lower()

    match = re.search(
        r"precision[_=-]?(\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)+)",
        name,
    )
    if match:
        values = re.findall(r"\d+(?:\.\d+)?", match.group(1))
        return "-".join(values)

    match = re.search(
        r"precision[_=-]?(\d+(?:\.\d+)?)",
        name,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)

    match = re.search(
        r"\bprecision\s*[:=]\s*(\d+(?:\.\d+)?)",
        content,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)

    return None


def normalize_observable(value):
    """Normalize observable labels."""
    value = value.strip().strip("_- ")
    value = re.sub(r"\s+", "", value)

    aliases = {
        "Z": "Z",
        "Z+ZZ": "Z+ZZ",
        "Z_ZZ": "Z+ZZ",
        "XYZ": "XYZ",
        "XYZ+XYZ-2body": "XYZ+XYZ-2body",
        "XYZ+XYZ-2body-all": "XYZ+XYZ-2body-all",
    }

    return aliases.get(value, value or None)


def extract_observable(filename, content):
    """Identify observable labels from known filename patterns."""
    name = Path(filename).stem

    # Remove backend prefix, including double underscores.
    name = re.sub(
        r"^(estimator-exact|shots|spinpulse)(?:__|_)",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = name.lstrip("_- ")

    # Handle obs-Z, obs-Z+ZZ, obs-XYZ, etc.
    match = re.search(
        r"obs-(.+?)(?=_precision|_seed|$)",
        name,
        re.IGNORECASE,
    )
    if match:
        return normalize_observable(match.group(1))

    # Handle filenames such as Z_seed2 and Z_precision0.05_seed2.
    name = re.sub(r"_precision.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_seed[_=-]?\d+.*$", "", name, flags=re.IGNORECASE)
    name = name.strip("_- ")

    # Handle observables_Z naming.
    name = re.sub(
        r"^observables?[_-]+",
        "",
        name,
        flags=re.IGNORECASE,
    )

    observable = normalize_observable(name)
    if observable:
        return observable

    # Fall back to observable labels in the log.
    patterns = [
        r"\bobservables?\s*[:=]\s*([A-Za-z0-9_+#-]+)",
        r"\bobs\s*[:=]\s*([A-Za-z0-9_+#-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            observable = normalize_observable(match.group(1))
            if observable:
                return observable

    return None


def identify_hamiltonian(path):
    """Identify the Hamiltonian from any parent directory or the filename."""
    known = [
        "tfim-longitudinal",
        "tfim-native",
        "native-exchange",
        "random",
        "tfim",
    ]

    # Check parent directories, closest first.
    # For observables/native-exchange/spinpulse__Z_seed2.txt,
    # this finds "native-exchange".
    for part in reversed(path.parts[:-1]):
        part_lower = part.lower().strip()
        for hamiltonian in known:
            if part_lower == hamiltonian:
                return hamiltonian

    # Fall back to searching the filename.
    name = path.stem.lower()
    for hamiltonian in known:
        if re.search(
            rf"(?<![a-z0-9]){re.escape(hamiltonian)}(?![a-z0-9])",
            name,
        ):
            return hamiltonian

    return None


# ------------------------------------------------------------------
# Uniform naming
# ------------------------------------------------------------------


def safe_label(value):
    """Convert a label into a filename-safe string."""
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", str(value)).strip("-")


def make_filename(config):
    """Build a uniform filename from the identified configuration."""
    parts = [
        config["backend"],
        config["hamiltonian"] or "unknown-hamiltonian",
        config["observable"] or "unknown-observable",
    ]

    if config["precision"] is not None:
        parts.append(f"precision{config['precision']}")

    if config["seed"] is not None:
        parts.append(f"seed{config['seed']}")
    else:
        parts.append("seed-unknown")

    return "_".join(safe_label(part) for part in parts) + ".txt"


def parse_config(path):
    """Read a log and infer its naming metadata."""
    try:
        content = path.read_text(errors="replace")
    except OSError as exc:
        return None, f"Could not read file: {exc}"

    config = {
        "backend": extract_backend(path.name, content),
        "hamiltonian": identify_hamiltonian(path),
        "observable": extract_observable(path.name, content),
        "precision": extract_precision(path.name, content),
        "seed": extract_seed(path.name, content),
    }

    return config, None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    if not ROOT.exists():
        print(f"ERROR: Source directory does not exist: {ROOT}")
        return

    # Exclude the destination directory if it is inside ROOT.
    files = sorted(
        path
        for path in ROOT.rglob("*.txt")
        if path.is_file() and DEST_ROOT not in path.parents
    )

    print(f"Source:      {ROOT}")
    print(f"Destination: {DEST_ROOT}")
    print(f"Mode:        {'DRY RUN' if DRY_RUN else 'COPY'}")
    print(f"Files found: {len(files)}\n")

    copied = 0
    skipped = 0
    errors = 0

    # Track destination names reserved during this run.
    reserved = set()

    for source in files:
        config, error = parse_config(source)

        if error:
            print(f"ERROR: {source}: {error}")
            errors += 1
            continue

        filename = make_filename(config)
        destination = DEST_ROOT / filename

        # Skip if the standardized destination already exists.
        # Also skip duplicate destination names from this run.
        if destination.exists() or destination in reserved:
            print("SKIP (destination already exists or is reserved):")
            print(f"  From: {source}")
            print(f"  To:   {destination}\n")
            skipped += 1
            continue

        reserved.add(destination)

        print(f"{'WOULD COPY' if DRY_RUN else 'COPY'}:")
        print(f"  From: {source}")
        print(f"  To:   {destination}")
        print(
            "  Config: "
            f"backend={config['backend']}, "
            f"H={config['hamiltonian']}, "
            f"observable={config['observable']}, "
            f"precision={config['precision']}, "
            f"seed={config['seed']}\n"
        )

        if not DRY_RUN:
            try:
                DEST_ROOT.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            except OSError as exc:
                print(f"  COPY FAILED: {exc}\n")
                errors += 1
                continue

        copied += 1

    print("=" * 60)
    print(f"Files found: {len(files)}")
    print(f"{'Would copy' if DRY_RUN else 'Copied'}: {copied}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    if DRY_RUN:
        print("\nReview the mappings, then set DRY_RUN = False to copy.")


if __name__ == "__main__":
    main()
