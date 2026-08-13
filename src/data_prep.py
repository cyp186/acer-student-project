from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["code_module", "code_presentation", "id_student"]
UCI_ZIP_NAME = "predict+students+dropout+and+academic+success.zip"
OULAD_ZIP_NAME = "open+university+learning+analytics+dataset.zip"


def _clean_column_name(name: str) -> str:
    name = name.strip().replace("Nacionality", "Nationality")  # typo in the UCI file
    name = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return name.lower()


def prepare_uci(uci_zip: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    with zipfile.ZipFile(uci_zip) as archive:
        frame = pd.read_csv(archive.open("data.csv"), sep=";")

    frame.columns = [_clean_column_name(col) for col in frame.columns]
    # still enrolled = outcome unknown, so drop those
    frame = frame.loc[frame["target"].isin(["Dropout", "Graduate"])].copy()
    frame["target_dropout"] = (frame["target"] == "Dropout").astype("int8")

    # curricular_units_* are semester results — after enrolment
    leakage_columns = [col for col in frame if col.startswith("curricular_units_")]
    frame = frame.drop(columns=["target", *leakage_columns])

    categorical_columns = [
        "marital_status",
        "application_mode",
        "course",
        "daytime_evening_attendance",
        "previous_qualification",
        "nationality",
        "mother_s_qualification",
        "father_s_qualification",
        "mother_s_occupation",
        "father_s_occupation",
        "displaced",
        "educational_special_needs",
        "debtor",
        "tuition_fees_up_to_date",
        "gender",
        "scholarship_holder",
        "international",
    ]
    numeric_columns = [
        "application_order",
        "previous_qualification_grade",
        "admission_grade",
        "age_at_enrollment",  # that's the name in the UCI file
        "unemployment_rate",
        "inflation_rate",
        "gdp",
    ]

    expected = set(categorical_columns + numeric_columns + ["target_dropout"])
    unexpected = set(frame.columns) - expected
    missing = expected - set(frame.columns)
    if unexpected or missing:
        raise ValueError(
            f"Unexpected UCI schema. Extra={sorted(unexpected)}, missing={sorted(missing)}"
        )

    return frame, categorical_columns, numeric_columns


def _aggregate_oulad_vle(
    archive: zipfile.ZipFile,
    vle_lookup: pd.DataFrame,
    start_day: int = 0,
    end_day: int = 30,
    chunksize: int = 750_000,
) -> pd.DataFrame:
    # studentVle is huge (~10m rows), so read it in chunks
    parts: list[pd.DataFrame] = []
    usecols = [*KEYS, "id_site", "date", "sum_click"]

    for chunk in pd.read_csv(
        archive.open("studentVle.csv"), usecols=usecols, chunksize=chunksize
    ):
        chunk = chunk.loc[chunk["date"].between(start_day, end_day)].copy()
        if chunk.empty:
            continue

        chunk = chunk.merge(
            vle_lookup[["id_site", "code_module", "code_presentation", "activity_type"]],
            on=["id_site", "code_module", "code_presentation"],
            how="left",
            validate="many_to_one",
        )
        chunk["activity_type"] = chunk["activity_type"].fillna("unknown")

        base = (
            chunk.groupby(KEYS, observed=True)
            .agg(
                total_clicks=("sum_click", "sum"),
                active_days=("date", "nunique"),
                unique_sites=("id_site", "nunique"),
            )
            .reset_index()
        )
        activity = (
            chunk.pivot_table(
                index=KEYS,
                columns="activity_type",
                values="sum_click",
                aggfunc="sum",
                fill_value=0,
                observed=True,
            )
            .add_prefix("clicks_")
            .reset_index()
        )
        parts.append(base.merge(activity, on=KEYS, how="left"))

    if not parts:
        raise ValueError("No OULAD VLE activity was found in the requested day window.")

    combined = pd.concat(parts, ignore_index=True).fillna(0)
    value_columns = [column for column in combined.columns if column not in KEYS]

    # same student can show up in more than one chunk
    summed = combined.groupby(KEYS, as_index=False)[
        [c for c in value_columns if c not in {"active_days", "unique_sites"}]
    ].sum()

    # nunique across chunks isn't additive, so redo those with sets
    day_sets: dict[tuple, set] = {}
    site_sets: dict[tuple, set] = {}
    for chunk in pd.read_csv(
        archive.open("studentVle.csv"), usecols=usecols, chunksize=chunksize
    ):
        chunk = chunk.loc[chunk["date"].between(start_day, end_day)]
        for key, group in chunk.groupby(KEYS, sort=False):
            day_sets.setdefault(key, set()).update(group["date"].unique())
            site_sets.setdefault(key, set()).update(group["id_site"].unique())

    distinct = pd.DataFrame(
        [
            (*key, len(day_sets[key]), len(site_sets.get(key, set())))
            for key in day_sets
        ],
        columns=[*KEYS, "active_days", "unique_sites"],
    )
    result = summed.merge(distinct, on=KEYS, how="outer").fillna(0)
    result["average_clicks_per_active_day"] = np.divide(
        result["total_clicks"],
        result["active_days"],
        out=np.zeros(len(result), dtype=float),
        where=result["active_days"].to_numpy() != 0,
    )
    return result


def prepare_oulad(
    oulad_zip: Path,
    start_day: int = 0,
    end_day: int = 30,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    with zipfile.ZipFile(oulad_zip) as archive:
        info = pd.read_csv(archive.open("studentInfo.csv"), na_values="?")
        registration = pd.read_csv(
            archive.open("studentRegistration.csv"), na_values="?"
        )
        vle_lookup = pd.read_csv(archive.open("vle.csv"), na_values="?")
        activity = _aggregate_oulad_vle(
            archive, vle_lookup, start_day=start_day, end_day=end_day
        )

    cohort = info.merge(registration, on=KEYS, how="inner", validate="one_to_one")

    # prediction point = end of day 30, so they have to be registered by then
    # and not already withdrawn
    cohort = cohort.loc[
        cohort["date_registration"].notna()
        & (cohort["date_registration"] <= end_day)
        & (
            cohort["date_unregistration"].isna()
            | (cohort["date_unregistration"] > end_day)
        )
    ].copy()
    cohort["target_withdrawn"] = (cohort["final_result"] == "Withdrawn").astype(
        "int8"
    )

    cohort = cohort.merge(activity, on=KEYS, how="left", validate="one_to_one")
    activity_columns = [
        col
        for col in cohort.columns
        if col.startswith("clicks_")
        or col
        in {
            "total_clicks",
            "active_days",
            "unique_sites",
            "average_clicks_per_active_day",
        }
    ]
    cohort[activity_columns] = cohort[activity_columns].fillna(0)
    cohort["no_vle_activity"] = (cohort["total_clicks"] == 0).astype("int8")

    categorical_columns = [
        "code_module",
        "code_presentation",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]
    numeric_columns = [
        "num_of_prev_attempts",
        "studied_credits",
        "date_registration",
        "no_vle_activity",
        *activity_columns,
    ]

    keep = [*KEYS, *categorical_columns, *numeric_columns, "target_withdrawn"]
    keep = list(dict.fromkeys(keep))
    cohort = cohort[keep].copy()
    return cohort, categorical_columns, numeric_columns


def build_processed_tables(
    uci_zip: Path, oulad_zip: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    uci, _, _ = prepare_uci(uci_zip)
    oulad, _, _ = prepare_oulad(oulad_zip)
    uci.to_csv(output_dir / "uci_enrolment_model.csv", index=False)
    oulad.to_csv(output_dir / "oulad_day30_model.csv", index=False)
    return uci, oulad


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the UCI and OULAD modelling tables.")
    parser.add_argument("--uci-zip", type=Path, required=True)
    parser.add_argument("--oulad-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    uci, oulad = build_processed_tables(args.uci_zip, args.oulad_zip, args.output_dir)
    print(
        f"UCI: {len(uci):,} rows; dropout rate={uci['target_dropout'].mean():.2%}"
    )
    print(
        "OULAD: "
        f"{len(oulad):,} rows; later-withdrawal rate="
        f"{oulad['target_withdrawn'].mean():.2%}"
    )


if __name__ == "__main__":
    main()
