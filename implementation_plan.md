Goal: Consolidate rust-numpy parity/gap docs into a single NumPy 2.4 report and add an enum parity audit (NumPy enums vs rust-numpy enums).

Changes:
- [NEW] rust-numpy/PARITY_NUMPY_2_4.md — consolidated parity/gap report (summary, FFI/export gap, checklist, roadmap, enum parity section).
- [NEW] rust-numpy/ENUM_PARITY_NUMPY_2_4.csv — NumPy 2.4 enum list with rust-numpy coverage status.
- [NEW] rust-numpy/scripts/enum_audit.py — generate NumPy enum inventory and compare against rust-numpy enums.
- [MODIFY] rust-numpy/scripts/inventory_scanner.py — de-duplicate function inventory output and optionally export enums for parity checks.
- [MODIFY] rust-numpy/INVENTORY.md — regenerated from updated scanner.
- [DELETE] rust-numpy/GAP_ANALYSIS.md — merged into consolidated report.
- [DELETE] rust-numpy/PARITY.md — merged into consolidated report.
- [DELETE] rust-numpy/PARITY_AND_GAP_ANALYSIS.md — merged into consolidated report.
- [DELETE] rust-numpy/THOROUGH_GAP_ANALYSIS.md — merged into consolidated report.
- [DELETE] rust-numpy/PARITY_AUDIT_CHECKLIST_NUMPY_2_4.md — merged into consolidated report.
- [DELETE] rust-numpy/PARITY_BASELINE_NUMPY_2_4.md — summary moved into consolidated report.

Verification:
- cd rust-numpy
- cargo fmt --check
- cargo clippy -- -D warnings
- cargo test
- python scripts/enum_audit.py
