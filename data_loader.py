import pandas as pd
import numpy as np
import zipfile
import re
from pathlib import Path

BASE = Path(__file__).parent

BEAUTY_KEYWORDS = [
    "BEAUTY", "SALON", "BARBER", "NAIL", "SPA",
    "COSMET", "HAIR", "LASH", "BROW", "WAXING"
]

RETAIL_KEYWORDS = [
    "RESTAURANT", "FOOD", "GROCERY", "RETAIL", "SHOP",
    "STORE", "CAFE", "COFFEE", "BAR", "TAVERN"
]

ACS_COLS = {
    "S1903_C01_001E": "TotalHouseholds",
    "S1903_C01_012E": "Age25_44",
    "S1903_C01_034E": "NonfamilyHouseholds",
    "S1903_C03_001E": "MedianIncome",
}


def _load_business_licenses() -> pd.DataFrame:
    path = BASE / "Business_Licenses_-_Current_Active_20260402.csv"
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["ZIP CODE"] = df["ZIP CODE"].str.strip().str[:5]
    df = df[df["ZIP CODE"].str.match(r"^606\d{2}$", na=False)]
    return df


def _load_beauty_counts(bl: pd.DataFrame) -> pd.DataFrame:
    mask = bl["BUSINESS ACTIVITY"].str.upper().str.contains(
        "|".join(BEAUTY_KEYWORDS), na=False
    )
    return bl[mask].groupby("ZIP CODE").size().reset_index(name="BeautyCount")


def _load_retail_counts(bl: pd.DataFrame) -> pd.DataFrame:
    mask = bl["BUSINESS ACTIVITY"].str.upper().str.contains(
        "|".join(RETAIL_KEYWORDS), na=False
    )
    return bl[mask].groupby("ZIP CODE").size().reset_index(name="RetailCount")


def _load_acs() -> pd.DataFrame:
    path = BASE / "ACSST5Y2024.S1903_2026-02-11T171455" / "ACSST5Y2024.S1903-Data.csv"
    df = pd.read_csv(path, header=0, skiprows=[1], dtype=str)
    cols_needed = ["NAME"] + list(ACS_COLS.keys())
    df = df[cols_needed].copy()
    df.rename(columns=ACS_COLS, inplace=True)
    df["ZIP CODE"] = df["NAME"].str.extract(r"(\d{5})")
    df = df.dropna(subset=["ZIP CODE"])
    for col in ACS_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["MedianIncome", "TotalHouseholds"])
    return df[["ZIP CODE"] + list(ACS_COLS.values())]


def _load_cta_rides() -> pd.DataFrame:
    rides_path = next(BASE.glob("CTA_-_Ridership_-_*_Daily_Totals_*.csv"), None)
    kmz_path   = BASE / "CTA_RailStations.kmz"
    if rides_path is None or not kmz_path.exists():
        return pd.DataFrame(columns=["ZIP CODE", "TotalAvgRides"])

    # Average daily rides per station (2023–2025)
    cta = pd.read_csv(rides_path)
    cta["date"]  = pd.to_datetime(cta["date"])
    cta["rides"] = cta["rides"].str.replace(",", "").astype(int)
    cta = cta[cta["date"].dt.year >= 2023]
    avg = cta.groupby("station_id")["rides"].mean().reset_index(name="AvgDailyRides")

    # Parse KMZ for station coords
    with zipfile.ZipFile(kmz_path, "r") as z:
        kml = z.read("doc.kml").decode("utf-8")
    ids    = re.findall(r"Station ID</td>\s*<td>(\d+)</td>", kml)
    coords = re.findall(r"<coordinates>\s*([-\d.]+),([-\d.]+)", kml)
    geo = pd.DataFrame({
        "station_id": [int(i) + 40000 for i in ids],
        "lon": [float(c[0]) for c in coords],
        "lat": [float(c[1]) for c in coords],
    }).merge(avg, on="station_id", how="left")

    # Map each station to nearest Chicago ZIP centroid
    try:
        import pgeocode
        nomi = pgeocode.Nominatim("us")
        chicago_zips = [str(z).zfill(5) for z in range(60601, 60662)]
        zip_info = (
            nomi.query_postal_code(chicago_zips)[["postal_code", "latitude", "longitude"]]
            .dropna()
            .reset_index(drop=True)
        )

        def nearest_zip(lat, lon):
            dlat = zip_info["latitude"] - lat
            dlon = zip_info["longitude"] - lon
            return zip_info.loc[(dlat**2 + dlon**2).idxmin(), "postal_code"]

        geo["ZIP CODE"] = geo.apply(lambda r: nearest_zip(r["lat"], r["lon"]), axis=1)
    except Exception:
        return pd.DataFrame(columns=["ZIP CODE", "TotalAvgRides"])

    return geo.groupby("ZIP CODE")["AvgDailyRides"].sum().reset_index(name="TotalAvgRides")


def _normalize(series: pd.Series, winsorize: bool = False) -> pd.Series:
    if winsorize:
        series = series.clip(upper=series.quantile(0.95))
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def load_data() -> pd.DataFrame:
    fast_path = BASE / "final_data.csv"
    if fast_path.exists():
        return pd.read_csv(fast_path, dtype={"ZIP CODE": str})

    bl      = _load_business_licenses()
    beauty  = _load_beauty_counts(bl)
    retail  = _load_retail_counts(bl)
    acs     = _load_acs()
    cta     = _load_cta_rides()

    df = acs.merge(beauty,  on="ZIP CODE", how="inner")
    df = df.merge(retail,   on="ZIP CODE", how="left")
    df = df.merge(cta,      on="ZIP CODE", how="left")

    df["BeautyCount"]  = df["BeautyCount"].fillna(0).astype(int)
    df["RetailCount"]  = df["RetailCount"].fillna(0)
    df["TotalAvgRides"] = df["TotalAvgRides"].fillna(0)

    # Derived metrics
    df["YoungShare"]     = df["Age25_44"]          / df["TotalHouseholds"]
    df["SingleShare"]    = df["NonfamilyHouseholds"] / df["TotalHouseholds"]
    df["BeautyPer1000"]  = df["BeautyCount"]  / df["TotalHouseholds"] * 1000
    df["RetailPer1000"]  = df["RetailCount"]  / df["TotalHouseholds"] * 1000
    df["RidesPer1000"]   = df["TotalAvgRides"] / df["TotalHouseholds"] * 1000

    # Opportunity Score (winsorize per-1000 metrics at p95 to reduce outlier distortion)
    df["_YoungRank"]   = _normalize(df["YoungShare"])
    df["_IncomeRank"]  = _normalize(df["MedianIncome"])
    df["_HHRank"]      = _normalize(df["TotalHouseholds"])
    df["_CompRank"]    = _normalize(df["BeautyPer1000"],  winsorize=True)
    df["_RetailRank"]  = _normalize(df["RetailPer1000"],  winsorize=True)

    df["OpportunityScore"] = (
        0.20 * df["_YoungRank"] +
        0.25 * df["_IncomeRank"] +
        0.10 * df["_HHRank"] +
        0.35 * df["_RetailRank"] -
        0.25 * df["_CompRank"]
    )
    df["OpportunityScore"] = _normalize(df["OpportunityScore"]) * 100

    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    return df.sort_values("OpportunityScore", ascending=False).reset_index(drop=True)
