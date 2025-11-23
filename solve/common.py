from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def _resolve_data_dir(user_dir: str | Path = "data") -> Path:
    """Find a data directory that actually contains the tariff txt files."""
    user_path = Path(user_dir)
    candidates = []
    if user_path.is_absolute():
        candidates.append(user_path)
        candidates.append(user_path.parent / "data")
        if len(user_path.parents) > 1:
            candidates.append(user_path.parents[1] / "data")
    else:
        candidates.extend(
            [
                user_path,
                Path.cwd() / user_path,
                Path.cwd().parent / user_path,
                Path(__file__).resolve().parent / user_path,
                Path(__file__).resolve().parent.parent / user_path,
                Path(__file__).resolve().parent.parent / "data",
            ]
        )
    for cand in list(dict.fromkeys(candidates)):  # deduplicate while preserving order
        if cand.exists() and list(cand.glob("*.txt")):
            return cand
    raise FileNotFoundError(
        f"Could not find data directory with txt files among: {[str(c) for c in candidates]}"
    )


def load_tariff_data(
    data_dir: str | Path = "data", force_sep: str | None = None
) -> pd.DataFrame:
    """
    Load all tariff text files in the data directory and return a combined DataFrame.
    The loader auto-detects separators, normalizes HTS codes to 8-digit strings,
    and adds a source_year column inferred from filenames.
    If the files are known to be CSV, pass force_sep=\",\" for a fast path.
    """
    data_dir = _resolve_data_dir(data_dir)
    frames: List[pd.DataFrame] = []
    for path in sorted(data_dir.glob("*.txt")):
        with path.open(encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        sep_guess = force_sep or ("," if first_line.count(",") >= first_line.count("|") else "|")

        def _try_read(encoding: str, sep):
            return pd.read_csv(
                path,
                sep=sep,
                low_memory=False,
                encoding=encoding,
                encoding_errors="ignore",
            )

        df = None
        for enc in ("utf-8", "latin1"):
            try:
                df = _try_read(enc, sep_guess)
                break
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
                try:
                    df = pd.read_csv(
                        path,
                        sep=None,
                        engine="python",
                        low_memory=False,
                        encoding=enc,
                        encoding_errors="ignore",
                    )
                    break
                except Exception:
                    continue
        if df is None:
            raise ValueError(f"Failed to parse {path}")
        df["source_file"] = path.name
        year_match = re.search(r"(20\\d{2}|201\\d)", path.stem)
        df["source_year"] = int(year_match.group(1)) if year_match else pd.NA
        # Normalize HTS codes to 8-digit strings for easier matching.
        df["hts8"] = (
            df["hts8"]
            .astype(str)
            .str.replace(r"\\.0$", "", regex=True)
            .str.replace(r"\\D", "", regex=True)
            .str.zfill(8)
        )
        # Numeric conversions for the main MFN ad valorem rate; fallback to ad val average.
        df["mfn_ad_val_rate"] = pd.to_numeric(df.get("mfn_ad_val_rate"), errors="coerce")
        df["mfn_ave"] = pd.to_numeric(df.get("mfn_ave"), errors="coerce")
        if "end_effective_date" in df.columns:
            df["end_effective_date"] = pd.to_datetime(
                df["end_effective_date"], errors="coerce"
            )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def filter_tariffs(df: pd.DataFrame, codes: Iterable[str]) -> pd.DataFrame:
    codes_normalized = {str(code).replace(".", "").zfill(8) for code in codes}
    return df[df["hts8"].isin(codes_normalized)].copy()


def simple_pass_through(
    base_price: float, tariff: float, pass_through: float = 0.7
) -> float:
    """Estimate post-tariff consumer price with a partial pass-through factor."""
    return base_price * (1 + tariff * pass_through)


def elasticity_response(volume: float, price_change: float, elasticity: float) -> float:
    """Apply constant elasticity response."""
    return volume * (1 + elasticity * price_change)


def laffer_revenue(
    import_value: float, rate: float, elasticity: float = -1.1
) -> float:
    """
    Stylized tariff revenue with import demand elasticity.
    import_value: baseline import value without tariff.
    rate: ad valorem tariff rate (e.g., 0.2 for 20%).
    """
    adjusted_import = import_value * (1 + elasticity * rate)
    adjusted_import = max(adjusted_import, 0)
    return adjusted_import * rate
