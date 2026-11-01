from pathlib import Path
import tempfile

import riasopy

with tempfile.TemporaryDirectory() as directory:
    gem = Path(directory) / "sample.gem"
    rows = ["geneID\tx\ty\tMIDCount"]
    for cell in range(20):
        for gene in range(10):
            rows.append(f"gene{gene}\t{cell * 100}\t0\t{1 + (cell + gene) % 9}")
    gem.write_text("\n".join(rows) + "\n", encoding="utf-8")
    data = riasopy.read_gem(str(gem), bin_size=100)
    data.cal_qc()
    png = data.plot_qc(kind="hist", metrics=["total_counts"], return_bytes=True)
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SystemExit("QC rendering did not return a PNG")
    print(f"RIASOPY {riasopy.__version__}: GEM loading and QC plotting passed ({len(png)} bytes)")
