import re
import shutil
from pathlib import Path

# Directories
SOURCE_ROOT = Path("results/benchmarks/ipc/observables")
DEST_DIR = Path("results/benchmarks/ipc/observables_copy")

# Hamiltonians
HAMILTONIANS = [
    "random",
    "tfim",
    "tfim-longitudinal",
    "tfim-native",
    "native-exchange",
]

TARGET_PRECISION = 0.01


def parse_source_filename(path):
    """
    Parse a SpinPulse filename and return:
        (observable, seed, precision)

    Recognized observables:
        Z       -> Z
        Z_ZZ    -> Z+ZZ

    Supports precision formats such as:
        precision001
        precision_001
        precision0.01
        precision_0.01

    Seed and precision can appear in either order.
    """

    stem = path.stem

    if not stem.startswith("spinpulse_"):
        return None

    # Check Z_ZZ first, otherwise it would match Z.
    if re.search(r"_Z_ZZ(?:_|$)", stem):
        observable = "Z+ZZ"
    elif re.search(r"_Z(?:_|$)", stem):
        observable = "Z"
    else:
        return None

    # Extract seed.
    seed_match = re.search(r"_seed(\d+)", stem)
    if not seed_match:
        return None

    seed = int(seed_match.group(1))

    # Extract precision.
    precision_match = re.search(r"_precision_?([^_]+)", stem)
    if not precision_match:
        return None

    precision_string = precision_match.group(1)

    # Handle precision001 as 0.01.
    if precision_string == "001":
        precision = 0.01
    else:
        try:
            precision = float(precision_string)
        except ValueError:
            return None

    return observable, seed, precision


def copy_files():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    overwritten = 0
    skipped_precision = 0
    unmatched = 0

    for hamiltonian in HAMILTONIANS:
        source_dir = SOURCE_ROOT / hamiltonian

        if not source_dir.exists():
            print(f"Missing directory: {source_dir}")
            continue

        # Search recursively for all .txt files.
        source_files = sorted(source_dir.rglob("*.txt"))

        for source_file in source_files:
            parsed = parse_source_filename(source_file)

            if parsed is None:
                unmatched += 1
                continue

            observable, seed, precision = parsed

            # Only copy precision 0.01.
            if abs(precision - TARGET_PRECISION) > 1e-12:
                skipped_precision += 1
                continue

            # Rename Z_ZZ to Z+ZZ in the destination filename.
            destination_name = (
                f"spinpulse_{hamiltonian}_{observable}"
                f"_seed{seed}_precision_0.01.txt"
            )

            destination_file = DEST_DIR / destination_name

            if destination_file.exists():
                overwritten += 1
            else:
                copied += 1

            shutil.copy2(source_file, destination_file)

            print(f"{source_file} -> {destination_file}")

    print("\nCopy summary")
    print("------------")
    print(f"Copied:              {copied}")
    print(f"Overwritten:         {overwritten}")
    print(f"Skipped (precision): {skipped_precision}")
    print(f"Unmatched:           {unmatched}")
    print(f"Destination:         {DEST_DIR}")


if __name__ == "__main__":
    copy_files()
