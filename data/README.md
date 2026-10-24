# Input data

The raw files are **not** in this repository: they are Bloomberg settlement data licensed for the author's own use.
Put your own copies in `data/raw/` (git-ignored) with exactly these names and columns and every script will run.
File names are set in `config.yaml` (`data:` section).

## `data/raw/sr3_settles.csv` - daily settlement prices, one row per contract per day

| column | type | units / example | notes |
|---|---|---|---|
| `date` | date | `2024-03-15` | trading session (exchange holidays simply absent) |
| `contract` | text | `SR3Z26` | 3-month SOFR future: `SR3` + CME month code + 2-digit year |
| `settle` | float | `96.125` | official settlement price (100 - rate). Rate in bp = (100 - settle) x 100 |
| `volume` | float | contracts | optional; may be blank |
| `open_interest` | float | contracts | optional; may be blank (only used if `data.min_open_interest` > 0) |

* Frequency: daily. Range used: May-2018 (SR3 listing) to the latest session.
* Serial months (F, G, J, K, N, Q, V, X) may be included; they are ignored - only quarterly H/M/U/Z contracts form the curve.
* Bloomberg equivalent: `PX_SETTLE`, `PX_VOLUME`, `OPEN_INT` of `SFR<month><year> Comdty`.

## `data/raw/sr3_contracts.csv` - one row per contract

| column | type | example | notes |
|---|---|---|---|
| `contract` | text | `SR3Z26` | must match `sr3_settles.csv` |
| `ref_month` | text | `2026-12` | first month of the reference quarter (the contract month) |
| `first_trade` | date | `2019-09-30` | informational |
| `last_trade` | date | `2027-03-16` | last trading day (end of the reference quarter). Decides the roll |
| `quarterly` | bool | `True` | informational; quarterly status is re-derived from `ref_month` |

## Files the code does not need (kept for related projects)

`daily_series.csv` and `fomc_decisions.csv` may sit in the same folder; this project ignores them. Policy regimes and
event dates used on charts are written out in `config.yaml`.

## `data/interim/`

Reserved for cached intermediate files; git-ignored.
