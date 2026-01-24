import csv
import enum
import importlib
import inspect
import re
from pathlib import Path

MODULES = [
    "numpy",
    "numpy.linalg",
    "numpy.fft",
    "numpy.random",
    "numpy.polynomial",
    "numpy.ma",
    "numpy.char",
    "numpy.testing",
    "numpy.typing",
]


def load_rust_enums(src_root: Path):
    enum_names = set()
    for path in src_root.rglob("*.rs"):
        content = path.read_text(encoding="utf-8")
        for name in re.findall(r"pub\s+enum\s+([A-Za-z_][A-Za-z0-9_]*)", content):
            enum_names.add(name)
    return enum_names


def iter_public_names(module):
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in dir(module) if not name.startswith("_")]
    return names


def collect_numpy_enums():
    items = []
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to import {module_name}: {exc}") from exc

        for name in iter_public_names(module):
            try:
                obj = getattr(module, name)
            except Exception:
                continue

            if not inspect.isclass(obj):
                continue

            try:
                if not issubclass(obj, enum.Enum):
                    continue
            except Exception:
                continue

            items.append(
                {
                    "module": module_name,
                    "enum": name,
                    "qualified_name": f"{module_name}.{name}",
                }
            )

    unique = {}
    for item in items:
        key = (item["module"], item["enum"])
        unique[key] = item
    return list(unique.values())


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "module",
                "enum",
                "qualified_name",
                "status",
                "rust_enum",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    rust_root = Path(__file__).resolve().parent.parent
    rust_src = rust_root / "src"
    output_csv = rust_root / "ENUM_PARITY_NUMPY_2_4.csv"

    import numpy as np

    numpy_version = np.__version__
    if not numpy_version.startswith("2.4"):
        print(f"Warning: numpy version {numpy_version} != 2.4.x")

    rust_enums = load_rust_enums(rust_src)
    numpy_enums = collect_numpy_enums()

    rows = []
    for item in sorted(numpy_enums, key=lambda r: (r["module"], r["enum"])):
        enum_name = item["enum"]
        status = "present" if enum_name in rust_enums else "missing"
        rows.append(
            {
                "module": item["module"],
                "enum": enum_name,
                "qualified_name": item["qualified_name"],
                "status": status,
                "rust_enum": enum_name if status == "present" else "",
                "notes": "",
            }
        )

    write_csv(rows, output_csv)

    total = len(rows)
    present = sum(1 for r in rows if r["status"] == "present")
    missing = total - present
    print(f"Wrote {output_csv}")
    print(f"NumPy version: {numpy_version}")
    print(f"Total NumPy enums: {total}")
    print(f"Present in rust-numpy: {present}")
    print(f"Missing in rust-numpy: {missing}")


if __name__ == "__main__":
    main()
