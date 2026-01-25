import argparse
import os
import re


def scan_rust_codebase(src_path):
    inventory = []
    seen = set()

    # Walk through the directory
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith(".rs"):
                file_path = os.path.join(root, file)
                module_name = (
                    os.path.relpath(file_path, src_path)
                    .replace("/", "::")
                    .replace(".rs", "")
                )
                if module_name.endswith("::mod"):
                    module_name = module_name[:-5]

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Regex to find pub fn
                    # This is a simple regex and might miss some edge cases or complex logic,
                    # but should be good enough for a rough inventory.
                    # It captures: pub fn function_name
                    matches = re.findall(r"pub\s+fn\s+([a-z_][a-z0-9_]*)", content)

                    for func_name in matches:
                        key = (module_name, func_name)
                        if key in seen:
                            continue
                        seen.add(key)
                        inventory.append(
                            {
                                "module": module_name,
                                "function": func_name,
                                "status": "Implemented",  # Assumed implemented if found
                            }
                        )

    return inventory


def scan_rust_enums(src_path):
    inventory = []
    seen = set()

    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith(".rs"):
                file_path = os.path.join(root, file)
                module_name = (
                    os.path.relpath(file_path, src_path)
                    .replace("/", "::")
                    .replace(".rs", "")
                )
                if module_name.endswith("::mod"):
                    module_name = module_name[:-5]

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    matches = re.findall(
                        r"pub\s+enum\s+([A-Za-z_][A-Za-z0-9_]*)", content
                    )
                    for enum_name in matches:
                        key = (module_name, enum_name)
                        if key in seen:
                            continue
                        seen.add(key)
                        inventory.append(
                            {
                                "module": module_name,
                                "enum": enum_name,
                                "status": "Implemented",
                            }
                        )

    return inventory


def generate_markdown(inventory):
    md = "# Rust-NumPy Function Inventory\n\n"
    md += "| Rust Module | Rust Function | Status | Notes |\n"
    md += "|---|---|---|---|\n"

    # Sort for better readability
    inventory.sort(key=lambda x: (x["module"], x["function"]))

    for item in inventory:
        md += f"| {item['module']} | {item['function']} | {item['status']} | |\n"

    return md


def generate_enum_markdown(inventory):
    md = "# Rust-NumPy Enum Inventory\n\n"
    md += "| Rust Module | Rust Enum | Status | Notes |\n"
    md += "|---|---|---|---|\n"

    inventory.sort(key=lambda x: (x["module"], x["enum"]))

    for item in inventory:
        md += f"| {item['module']} | {item['enum']} | {item['status']} | |\n"

    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan rust-numpy for public APIs.")
    parser.add_argument(
        "--include-enums",
        action="store_true",
        help="Include a Rust enum inventory section.",
    )
    args = parser.parse_args()

    src_dir = "./src"
    inventory = scan_rust_codebase(src_dir)
    markdown_output = generate_markdown(inventory)

    if args.include_enums:
        enum_inventory = scan_rust_enums(src_dir)
        markdown_output += "\n" + generate_enum_markdown(enum_inventory)

    # Print to stdout so we can capture it
    print(markdown_output)
