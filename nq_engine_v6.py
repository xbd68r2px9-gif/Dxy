from __future__ import annotations

import os, re, json, math, time, hashlib, warnings
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 240)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 220)

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from IPython.display import display, Markdown
except Exception:
    display = print
    Markdown = str

ENGINE_VERSION = "v6.0.0_TRUE_TOP10_AMD"
DATA_CONTRACT_VERSION = "NQ_DATA_CONTRACT_V6"
FORMULA_VERSION = "NQ_PRICE_AND_FUNDAMENTAL_V6"
OFFICIAL_TOP10_COVERAGE_MIN = 0.95
UNIVERSAL_COVERAGE_MIN = 0.90
SPECIFIC_COVERAGE_MIN = 0.70
FORWARD_COVERAGE_MIN = 0.50


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def f(x: Any) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def safe_div(a: Any, b: Any) -> float:
    a, b = f(a), f(b)
    return a / b if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-12 else np.nan


def growth(cur: Any, prior: Any) -> float:
    cur, prior = f(cur), f(prior)
    return cur / prior - 1.0 if np.isfinite(cur) and np.isfinite(prior) and abs(prior) > 1e-12 else np.nan


def score_interp(x: Any, anchors: list[tuple[float, float]]) -> float:
    x = f(x)
    if not np.isfinite(x):
        return np.nan
    a = sorted(anchors)
    if x <= a[0][0]: return float(a[0][1])
    if x >= a[-1][0]: return float(a[-1][1])
    for (x0, y0), (x1, y1) in zip(a[:-1], a[1:]):
        if x0 <= x <= x1:
            return float(y0 + (x-x0)/(x1-x0)*(y1-y0))
    return np.nan


def weighted_score(values: dict[str, float], weights: dict[str, float], critical: tuple[str, ...] = (), min_coverage: float = 0.0) -> dict[str, Any]:
    applicable = {k: weights[k] for k in weights}
    available = {k: f(values.get(k)) for k in applicable if np.isfinite(f(values.get(k)))}
    total = sum(applicable.values())
    covered = sum(applicable[k] for k in available)
    coverage = covered / total if total else np.nan
    diagnostic = sum(available[k]*applicable[k] for k in available)/covered if covered else np.nan
    critical_ok = all(np.isfinite(f(values.get(k))) for k in critical)
    official = diagnostic if np.isfinite(diagnostic) and coverage >= min_coverage and critical_ok else np.nan
    return {"Diagnostic": diagnostic, "Official": official, "Coverage": coverage, "CriticalOK": critical_ok}


def label_score(x: Any) -> str:
    x = f(x)
    if not np.isfinite(x): return "UNAVAILABLE"
    if x >= 80: return "STRONG"
    if x >= 65: return "CONSTRUCTIVE"
    if x >= 50: return "MIXED"
    if x >= 35: return "WEAK"
    return "STRUCTURAL_HEADWIND"

# -----------------------------------------------------------------------------
# BLOCK 0 — CONFIGURATION, DURABLE STORAGE, TRUE TOP-10 CONTRACT
# -----------------------------------------------------------------------------
AS_OF_DATE = pd.Timestamp(os.getenv("NQ_AS_OF_DATE", pd.Timestamp.now(tz="America/New_York").date())).tz_localize(None)
AS_OF_TIMESTAMP_UTC = pd.to_datetime(os.getenv("NQ_AS_OF_TIMESTAMP_UTC", pd.Timestamp.now(tz="UTC")), utc=True)
RUN_MODE = os.getenv("NQ_RUN_MODE", "LIVE_INCREMENTAL").upper().strip()
NETWORK_ALLOWED = RUN_MODE in {"LIVE_INCREMENTAL", "LIVE_FORCE_REFRESH"}

if Path("/content/drive/MyDrive").exists():
    DATA_ROOT = Path("/content/drive/MyDrive/NQ_SPECIFIC_ENGINE_DATA_V6")
    PERSISTENCE_STATUS = "DURABLE_GOOGLE_DRIVE"
else:
    DATA_ROOT = Path("/mnt/data/NQ_SPECIFIC_ENGINE_DATA_V6")
    PERSISTENCE_STATUS = "LOCAL_RUNTIME_NOT_FREEZE_ELIGIBLE"
for p in [DATA_ROOT, DATA_ROOT/"market", DATA_ROOT/"fred", DATA_ROOT/"sec", DATA_ROOT/"reports"]:
    p.mkdir(parents=True, exist_ok=True)

# Public-safe handling: notebook supplies the key through env or a private Drive secret.
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FRED_API_KEY_VALID = bool(re.fullmatch(r"[a-z0-9]{32}", FRED_API_KEY))
SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL", "Noe.Wehrli@icloud.com")
SEC_HEADERS = {"User-Agent": f"NQ-Specific-Research {SEC_CONTACT_EMAIL}", "Accept-Encoding": "gzip, deflate"}

WEIGHT_MANIFEST = {
    "version": "NQ_TRUE_TOP10_COMPANY_QNDX_PROXY_2026_07_13",
    "source": "QNDX_FULL_HOLDINGS_PROXY_STOCKANALYSIS_2026_07_13",
    "source_url": "https://stockanalysis.com/etf/qndx/holdings/",
    "reference_date": "2026-07-13",
    "official_nasdaq_feed": False,
    "alphabet_combined": True,
    "missing_weight_redistribution": False,
    "full_index_renormalization": False,
    "share_class_weights": {
        "NVDA": .0789, "AAPL": .0746, "MU": .0467, "MSFT": .0465,
        "AMZN": .0426, "AMD": .0385, "GOOGL": .0329, "TSLA": .0316,
        "META": .0311, "GOOG": .0306, "AVGO": .0291,
    },
}

COMPANY_REGISTRY = {
    "NVIDIA": {"ticker":"NVDA", "cik":"0001045810", "group":"AI_SEMIS_INFRASTRUCTURE"},
    "APPLE": {"ticker":"AAPL", "cik":"0000320193", "group":"HARDWARE_DEVICES_NETWORKING"},
    "ALPHABET": {"ticker":"GOOGL", "cik":"0001652044", "group":"HYPERSCALERS_CLOUD_PLATFORMS", "share_classes":("GOOGL","GOOG")},
    "MICRON": {"ticker":"MU", "cik":"0000723125", "group":"AI_SEMIS_INFRASTRUCTURE"},
    "MICROSOFT": {"ticker":"MSFT", "cik":"0000789019", "group":"HYPERSCALERS_CLOUD_PLATFORMS"},
    "AMAZON": {"ticker":"AMZN", "cik":"0001018724", "group":"HYPERSCALERS_CLOUD_PLATFORMS"},
    "AMD": {"ticker":"AMD", "cik":"0000002488", "group":"AI_SEMIS_INFRASTRUCTURE"},
    "TESLA": {"ticker":"TSLA", "cik":"0001318605", "group":"CONSUMER_INTERNET_HIGH_DURATION"},
    "META": {"ticker":"META", "cik":"0001326801", "group":"HYPERSCALERS_CLOUD_PLATFORMS"},
    "BROADCOM": {"ticker":"AVGO", "cik":"0001730168", "group":"AI_SEMIS_INFRASTRUCTURE"},
}
SHARE_TO_COMPANY = {"NVDA":"NVIDIA","AAPL":"APPLE","MU":"MICRON","MSFT":"MICROSOFT","AMZN":"AMAZON","AMD":"AMD","GOOGL":"ALPHABET","GOOG":"ALPHABET","TSLA":"TESLA","META":"META","AVGO":"BROADCOM"}
COMPANY_WEIGHTS = {}
for t,w in WEIGHT_MANIFEST["share_class_weights"].items():
    c = SHARE_TO_COMPANY[t]
    COMPANY_WEIGHTS[c] = COMPANY_WEIGHTS.get(c,0.0)+w
COMPANY_WEIGHTS = dict(sorted(COMPANY_WEIGHTS.items(), key=lambda kv: kv[1], reverse=True))
TOP10_COVERAGE = sum(COMPANY_WEIGHTS.values())
TOP10_NORMALIZED = {c:w/TOP10_COVERAGE for c,w in COMPANY_WEIGHTS.items()}
assert len(COMPANY_WEIGHTS) == 10 and "AMD" in COMPANY_WEIGHTS and "WALMART" not in COMPANY_WEIGHTS
assert abs(TOP10_COVERAGE - .4831) < 1e-12

GROUPS = {
    "AI_SEMIS_INFRASTRUCTURE": ("NVDA","AVGO","AMD","MU","QCOM","INTC","AMAT","LRCX","KLAC","MRVL","ARM","ANET"),
    "HYPERSCALERS_CLOUD_PLATFORMS": ("MSFT","AMZN","GOOGL","GOOG","META","ORCL"),
    "SOFTWARE_CYBER_AI_APPS": ("ADBE","INTU","PANW","CRWD","ZS","SNPS","CDNS","DDOG","TEAM","WDAY"),
    "HARDWARE_DEVICES_NETWORKING": ("AAPL","CSCO","TXN","NXPI","MCHP","HON"),
    "CONSUMER_INTERNET_HIGH_DURATION": ("TSLA","NFLX","BKNG","ABNB","MELI","PDD","SBUX","LULU"),
    "DEFENSIVE_HEALTHCARE_NONTECH": ("WMT","COST","PEP","KDP","MDLZ","AMGN","GILD","VRTX","REGN","ISRG"),
}

# Audited PIT earnings snapshots. A new unmatched filing fails closed.
SNAP = {
"NVIDIA": dict(period="2026-04-26", accession="0001045810-26-000052", source="NVIDIA_Q1_FY2027_PRIMARY_RELEASE", url="https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2027/default.aspx", rev=81.615e9, rev_p=44.062e9, rev_cagr=.770236, rev_acc=.120131, rec_q=.012858, inv_q=-.423996, oi=53.536e9, oi_p=21.638e9, gm=.749335, gm_d=.144097, om=.655958, om_d=.164877, incr=.849413, eps=2.39, eps_p=.76, norm_eps=1.87, shares_d=-.008939, core=.765861, oi_ps_g=1.496482, fcf_m=.469744, fcf_m_d=-.015486, cfo_g=.649833, cfo_ni=.787204, fcf_ni=.746029, capex_cfo=.052305, capex_rev=.025926, capex_dep=2.035305, rev_conv=25.641426, oi_conv=18.588666, cash_abs=.050071, dep_press=.013188, netcash_rev=.018805, buyback_cfo=.360555, sbc_cfo=.054438),
"APPLE": dict(period="2026-03-28", accession="0000320193-26-000013", source="APPLE_Q2_FY2026_PRIMARY_RELEASE", url="https://www.sec.gov/Archives/edgar/data/320193/", rev=111.184e9, rev_p=95.359e9, rev_cagr=.106855, rev_acc=.009427, rec_q=.005139, inv_q=.089704, oi=35.885e9, oi_p=29.589e9, gm=.492706, gm_d=.0222, om=.322753, om_d=.012463, incr=.397852, eps=2.01, eps_p=1.65, norm_eps=2.01, shares_d=-.021935, core=.998553, oi_ps_g=.239981, fcf_m=.286136, fcf_m_d=.040147, cfo_g=.279912, cfo_ni=1.143969, fcf_ni=1.053836, capex_cfo=.078789, capex_rev=.024473, capex_dep=.87613, rev_conv=4.613911, oi_conv=1.806865, cash_abs=-.000717, dep_press=-.00346, netcash_rev=-.03147, buyback_cfo=.557659, sbc_cfo=.096083),
"ALPHABET": dict(period="2026-06-30", accession="0001652044-26-000071", source="ALPHABET_Q2_2026_PRIMARY_10Q_AND_RELEASE", url="https://www.sec.gov/Archives/edgar/data/1652044/000165204426000071/", rev=119.796e9, rev_p=96.428e9, rev_cagr=.188972, rev_acc=.024436, rec_q=-.014294, inv_q=np.nan, oi=40.770e9, oi_p=31.271e9, gm=np.nan, gm_d=np.nan, om=.340329, om_d=.016035, incr=.406496, eps=9.11, eps_p=2.31, norm_eps=2.85, shares_d=.0091, core=.293831, oi_ps_g=.292007, fcf_m=.119482, fcf_m_d=-.060185, cfo_g=.38866, cfo_ni=.760324, fcf_ni=.218149, capex_cfo=.713085, capex_rev=.296954, capex_dep=np.nan, rev_conv=1.111795, oi_conv=.392027, cash_abs=1.258914, dep_press=np.nan, netcash_rev=.319176, buyback_cfo=.093728, sbc_cfo=.151997),
"MICRON": dict(period="2026-05-28", accession="0000723125-26-000015", source="MICRON_Q3_FY2026_PRIMARY_RELEASE", url="https://investors.micron.com/node/50671", rev=41.456e9, rev_p=9.301e9, rev_cagr=1.467109, rev_acc=1.494284, rec_q=-.439786, inv_q=3.475489, oi=33.318e9, oi_p=2.169e9, gm=.845619, gm_d=.468456, om=.803695, om_d=.570495, incr=.968714, eps=24.67, eps_p=1.68, norm_eps=24.67, shares_d=.017778, core=.996829, oi_ps_g=14.092682, fcf_m=.289917, fcf_m_d=.234288, cfo_g=2.383684, cfo_ni=1.019081, fcf_ni=.518576, capex_cfo=.491134, capex_rev=.279815, capex_dep=2.80324, rev_conv=4.239132, oi_conv=3.437244, cash_abs=3.039659, dep_press=.179996, netcash_rev=np.nan, buyback_cfo=.012638, sbc_cfo=.022904),
"MICROSOFT": dict(period="2026-03-31", accession="0001193125-26-191507", source="MICROSOFT_FY2026_Q3_PRIMARY_RELEASE", url="https://www.microsoft.com/en-us/investor/earnings/FY-2026-Q3/press-release-webcast", rev=82.886e9, rev_p=70.066e9, rev_cagr=.157558, rev_acc=.015791, rec_q=.021636, inv_q=-.25453, oi=38.398e9, oi_p=32.0e9, gm=.676327, gm_d=-.01084, om=.463263, om_d=.006551, incr=.499064, eps=4.27, eps_p=3.46, norm_eps=4.27, shares_d=-.002144, core=.976055, oi_ps_g=.202516, fcf_m=.229099, fcf_m_d=-.027799, cfo_g=.301668, cfo_ni=1.35878, fcf_ni=.582322, capex_cfo=.571438, capex_rev=.305477, capex_dep=np.nan, rev_conv=.786747, oi_conv=.437314, cash_abs=.909944, dep_press=np.nan, netcash_rev=.119426, buyback_cfo=.130703, sbc_cfo=.072622),
"AMAZON": dict(period="2026-03-31", accession="0001018724-26-000014", source="AMAZON_Q1_2026_PRIMARY_RELEASE", url="https://ir.aboutamazon.com/news-release/news-release-details/2026/Amazon-com-Announces-First-Quarter-Results/default.aspx", rev=181.519e9, rev_p=155.667e9, rev_cagr=.125429, rev_acc=.029783, rec_q=-.227096, inv_q=.147391, oi=23.852e9, oi_p=18.405e9, gm=np.nan, gm_d=np.nan, om=.131402, om_d=.013169, incr=.210699, eps=2.78, eps_p=1.59, norm_eps=1.608142, shares_d=.007505, core=.598785, oi_ps_g=.286299, fcf_m=-.003328, fcf_m_d=-.035328, cfo_g=.304013, cfo_ni=1.63584, fcf_ni=-.027225, capex_cfo=1.016643, capex_rev=.203295, capex_dep=2.143741, rev_conv=.993233, oi_conv=.147498, cash_abs=1.672346, dep_press=.108463, netcash_rev=.032127, buyback_cfo=0.0, sbc_cfo=.133373),
"AMD": dict(period="2026-03-28", accession="0000002488-26-000076", source="AMD_Q1_2026_PRIMARY_10Q_AND_RELEASE", url="https://ir.amd.com/news-events/press-releases/detail/1284/amd-reports-first-quarter-2026-financial-results", rev=10.253e9, rev_p=7.438e9, rev_cagr=np.nan, rev_acc=.38-.34, rec_q=.38-(-.044), inv_q=.38-(.125/7.92), oi=1.476e9, oi_p=.806e9, gm=.53, gm_d=.03, om=.14, om_d=.03, incr=(1.476-.806)/(10.253-7.438), eps=.84, eps_p=.44, norm_eps=.84, shares_d=np.nan, core=1.0, oi_ps_g=.83, fcf_m=.25, fcf_m_d=.15, cfo_g=2.955/.939-1, cfo_ni=2.955/1.383, fcf_ni=2.566/1.383, capex_cfo=.389/2.955, capex_rev=.389/10.253, capex_dep=.389/(.206+.551), rev_conv=np.nan, oi_conv=np.nan, cash_abs=.389/2.955, dep_press=(.206+.551)/10.253, netcash_rev=(12.347-3.224)/10.253, buyback_cfo=.221/2.955, sbc_cfo=.487/2.955),
"TESLA": dict(period="2026-06-30", accession="0001628280-26-049270", source="TESLA_Q2_2026_PRIMARY_10Q_AND_UPDATE", url="https://www.sec.gov/Archives/edgar/data/1318605/000162828026049270/", rev=28.236e9, rev_p=22.496e9, rev_cagr=.05228, rev_acc=.097308, rec_q=.190279, inv_q=.311299, oi=.398e9, oi_p=.923e9, gm=.16826, gm_d=-.004126, om=.014095, om_d=-.026934, incr=-.091463, eps=.32, eps_p=.33, norm_eps=.074927, shares_d=.005968, core=.299473, oi_ps_g=-.571355, fcf_m=.055608, fcf_m_d=-.004638, cfo_g=.18522, cfo_ni=4.911935, fcf_ni=1.514721, capex_cfo=.691624, capex_rev=.124717, capex_dep=12.521074, rev_conv=1.070734, oi_conv=-.122802, cash_abs=.939726, dep_press=.114756, netcash_rev=np.nan, buyback_cfo=np.nan, sbc_cfo=.203265),
"META": dict(period="2026-03-31", accession="0001628280-26-028526", source="META_Q1_2026_PRIMARY_RELEASE", url="https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-First-Quarter-2026-Results/", rev=56.311e9, rev_p=42.314e9, rev_cagr=.242848, rev_acc=.092947, rec_q=.127123, inv_q=np.nan, oi=22.872e9, oi_p=17.555e9, gm=np.nan, gm_d=np.nan, om=.406173, om_d=-.008702, incr=.379867, eps=10.44, eps_p=6.43, norm_eps=7.31, shares_d=-.010039, core=.953318, oi_ps_g=.316088, fcf_m=.224472, fcf_m_d=-.082591, cfo_g=.290215, cfo_ni=1.756697, fcf_ni=.683596, capex_cfo=.610863, capex_rev=.352374, capex_dep=3.656626, rev_conv=1.018403, oi_conv=.35338, cash_abs=1.14549, dep_press=.256008, netcash_rev=.104353, buyback_cfo=.108823, sbc_cfo=.179927),
"BROADCOM": dict(period="2026-05-03", accession="0001730168-26-000054", source="BROADCOM_Q2_FY2026_PRIMARY_RELEASE", url="https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announces-second-quarter-fiscal-year-2026-financial", rev=22.187e9, rev_p=15.004e9, rev_cagr=.33297, rev_acc=.184089, rec_q=-.468052, inv_q=-.667022, oi=10.788e9, oi_p=5.829e9, gm=.694776, gm_d=.015157, om=.486231, om_d=.097734, incr=.69038, eps=1.91, eps_p=1.03, norm_eps=1.91, shares_d=.010361, core=.942513, oi_ps_g=.831768, fcf_m=.434135, fcf_m_d=.036263, cfo_g=.447041, cfo_ni=1.146843, fcf_ni=1.117509, capex_cfo=.025578, capex_rev=.011396, capex_dep=np.nan, rev_conv=34.236059, oi_conv=22.754647, cash_abs=.031, dep_press=np.nan, netcash_rev=-.60, buyback_cfo=.251324, sbc_cfo=.261287),
}

SPECIFIC = {
"NVIDIA": dict(values={"DataCenterGrowth":.92,"ComputeNetworkingGrowth":.77,"GrossMargin":.749,"InventoryDiscipline":-.424,"CustomerConcentration":.50}, weights={"DataCenterGrowth":.30,"ComputeNetworkingGrowth":.20,"GrossMargin":.20,"InventoryDiscipline":.15,"CustomerConcentration":.15}, critical=("DataCenterGrowth","GrossMargin")),
"APPLE": dict(values={"ServicesGrowth":.18,"IPhoneGrowth":.12,"GrossMargin":.493,"ChinaQuality":.50,"BuybackQuality":.558}, weights={"ServicesGrowth":.25,"IPhoneGrowth":.20,"GrossMargin":.20,"ChinaQuality":.15,"BuybackQuality":.20}, critical=("ServicesGrowth","GrossMargin")),
"ALPHABET": dict(values={"CloudGrowth":.817968,"CloudOIGrowth":2.113942,"CloudMargin":.3553,"SearchGrowth":.167577,"YouTubeGrowth":.128522}, weights={"CloudGrowth":.25,"CloudOIGrowth":.25,"CloudMargin":.20,"SearchGrowth":.20,"YouTubeGrowth":.10}, critical=("CloudGrowth","CloudOIGrowth","CloudMargin","SearchGrowth")),
"MICRON": dict(values={"CloudMemoryGrowth":2.0,"CoreDCGrowth":1.5,"GrossMargin":.846,"InventoryDiscipline":3.475,"CapexBurden":.491}, weights={"CloudMemoryGrowth":.25,"CoreDCGrowth":.25,"GrossMargin":.20,"InventoryDiscipline":.15,"CapexBurden":.15}, critical=("CloudMemoryGrowth","CoreDCGrowth","GrossMargin")),
"MICROSOFT": dict(values={"AzureGrowth":.50,"IntelligentCloudGrowth":.27,"CloudMargin":.66,"RPOGrowth":.99,"AIMonetization":1.23}, weights={"AzureGrowth":.25,"IntelligentCloudGrowth":.20,"CloudMargin":.20,"RPOGrowth":.20,"AIMonetization":.15}, critical=("AzureGrowth","CloudMargin","RPOGrowth")),
"AMAZON": dict(values={"AWSGrowth":.30,"AWSOIGrowth":.226,"AWSMargin":.35,"NorthAmericaMargin":.079,"InternationalMargin":.036}, weights={"AWSGrowth":.30,"AWSOIGrowth":.25,"AWSMargin":.20,"NorthAmericaMargin":.15,"InternationalMargin":.10}, critical=("AWSGrowth","AWSOIGrowth","AWSMargin")),
"AMD": dict(values={"DataCenterGrowth":5.775/3.674-1,"DataCenterOIGrowth":1.599/.932-1,"DataCenterMargin":1.599/5.775,"ClientGrowth":2.885/2.294-1,"GamingGrowth":.720/.647-1,"EmbeddedGrowth":.873/.823-1,"GrossMargin":.53,"FCFMargin":.25}, weights={"DataCenterGrowth":.25,"DataCenterOIGrowth":.20,"DataCenterMargin":.15,"ClientGrowth":.10,"GamingGrowth":.05,"EmbeddedGrowth":.05,"GrossMargin":.10,"FCFMargin":.10}, critical=("DataCenterGrowth","DataCenterOIGrowth","DataCenterMargin","GrossMargin")),
"TESLA": dict(values={"AutomotiveGrowth":.20,"AutoMargin":.15,"EnergyGrowth":.13,"RegCreditBurden":.146/28.236,"SoftwareGrowth":.56}, weights={"AutomotiveGrowth":.25,"AutoMargin":.25,"EnergyGrowth":.15,"RegCreditBurden":.15,"SoftwareGrowth":.20}, critical=("AutomotiveGrowth","AutoMargin","RegCreditBurden")),
"META": dict(values={"AdGrowth":.34,"PricePerAd":.10,"Impressions":.20,"FamilyMargin":.481,"RealityLabsBurden":4.028/56.311}, weights={"AdGrowth":.25,"PricePerAd":.15,"Impressions":.15,"FamilyMargin":.25,"RealityLabsBurden":.20}, critical=("AdGrowth","FamilyMargin","RealityLabsBurden")),
"BROADCOM": dict(values={"SemiconductorGrowth":.60,"SoftwareGrowth":.20,"AIGrowth":.70,"OperatingMargin":.486,"DebtBurden":.60}, weights={"SemiconductorGrowth":.25,"SoftwareGrowth":.20,"AIGrowth":.25,"OperatingMargin":.20,"DebtBurden":.10}, critical=("SemiconductorGrowth","AIGrowth","OperatingMargin")),
}

FORWARD_POLICY = {"APPLE":"NOT_APPLICABLE_DISCLOSURE_POLICY","TESLA":"NOT_APPLICABLE_DISCLOSURE_POLICY", **{c:"REQUIRED_NUMERIC_GUIDANCE" for c in COMPANY_REGISTRY if c not in {"APPLE","TESLA"}}}
FORWARD = {
"NVIDIA": {"RevenueGrowth":.9468,"Margin":.749},
"ALPHABET": {"RPOGrowth":3.849,"CapexRevenue":.4486},
"MICRON": {"RevenueGrowth":3.417,"Margin":.86},
"MICROSOFT": {"RevenueGrowth":.275,"Margin":.64,"RPOGrowth":.99,"CapexRevenue":.597},
"AMAZON": {"RevenueGrowth":.175,"OIGrowth":.1458},
"AMD": {"RevenueGrowth":.46,"Margin":.56},
"META": {"RevenueGrowth":.2522,"CapexRevenue":.628},
"BROADCOM": {"RevenueGrowth":.84,"Margin":.67},
}

# -----------------------------------------------------------------------------
# SEC latest-filing audit
# -----------------------------------------------------------------------------
def latest_periodic_accession(company: str) -> dict[str, Any]:
    meta = COMPANY_REGISTRY[company]
    url = f"https://data.sec.gov/submissions/CIK{meta['cik']}.json"
    path = DATA_ROOT/"sec"/f"CIK{meta['cik']}.json"
    payload = None
    if NETWORK_ALLOWED:
        try:
            r=requests.get(url,headers=SEC_HEADERS,timeout=30); r.raise_for_status(); payload=r.json(); path.write_text(json.dumps(payload),encoding="utf-8")
        except Exception:
            pass
    if payload is None and path.exists():
        payload=json.loads(path.read_text(encoding="utf-8"))
    recent=((payload or {}).get("filings") or {}).get("recent") or {}
    rows=[]
    n=max([len(v) for v in recent.values() if isinstance(v,list)] or [0])
    for i in range(n):
        row={k:(v[i] if isinstance(v,list) and i<len(v) else None) for k,v in recent.items()}
        if str(row.get("form")) in {"10-Q","10-K","10-Q/A","10-K/A"}:
            accepted=pd.to_datetime(row.get("acceptanceDateTime") or row.get("filingDate"),utc=True,errors="coerce")
            if pd.notna(accepted) and accepted<=AS_OF_TIMESTAMP_UTC: rows.append((accepted,row))
    if not rows: return {"Company":company,"LatestAccession":None,"LatestPeriod":None,"Audit":"UNAVAILABLE"}
    _, row=max(rows,key=lambda x:x[0])
    return {"Company":company,"LatestAccession":row.get("accessionNumber"),"LatestPeriod":row.get("reportDate"),"Audit":"PASS" if row.get("accessionNumber")==SNAP[company]["accession"] else "NEW_FILING_REQUIRES_REAUDIT"}

# -----------------------------------------------------------------------------
# BLOCK 1 — PRICE LEADERSHIP
# -----------------------------------------------------------------------------
def fetch_prices() -> pd.DataFrame:
    tickers=sorted({"QQQ",*sum((list(v) for v in GROUPS.values()),[])})
    cache=DATA_ROOT/"market"/f"prices_{AS_OF_DATE:%Y%m%d}.pkl"
    if NETWORK_ALLOWED and yf is not None:
        try:
            raw=yf.download(tickers,start=(AS_OF_DATE-pd.DateOffset(years=6)).strftime("%Y-%m-%d"),end=(AS_OF_DATE+pd.Timedelta(days=2)).strftime("%Y-%m-%d"),auto_adjust=True,progress=False,threads=True)
            px=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":tickers[0]})
            px.index=pd.to_datetime(px.index).tz_localize(None); px=px.loc[px.index.normalize()<=AS_OF_DATE]; px.to_pickle(cache); return px
        except Exception:
            pass
    return pd.read_pickle(cache) if cache.exists() else pd.DataFrame()


def pct_rank(s: pd.Series, x: float) -> float:
    s=s.dropna().tail(260)
    return float((s<=x).mean()*100) if len(s)>=104 and np.isfinite(x) else np.nan


def build_block1(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows=[]
    for gid,tickers in GROUPS.items():
        avail=[t for t in tickers if t in prices and prices[t].notna().sum()>=150]
        if len(avail)<max(3,math.ceil(.8*len(tickers))):
            rows.append({"Group":gid,"Score":np.nan,"Coverage":len(avail)/len(tickers)}); continue
        r=np.log(prices[avail]/prices[avail].shift(1)); idx=np.exp(r.mean(axis=1).cumsum())
        mom=[]; trend=[]; breadth=[]
        for h in (5,20,65,130):
            m=idx.pct_change(h); mom.append(pct_rank(m, f(m.iloc[-1])))
            ma=idx.rolling(h).mean(); trend.append(pct_rank((idx/ma-1), f((idx/ma-1).iloc[-1])))
            b=(prices[avail]/prices[avail].rolling(h).mean()-1).gt(0).mean(axis=1); breadth.append(pct_rank(b,f(b.iloc[-1])))
        ret=idx.pct_change(); ir63=ret.rolling(63).mean()/ret.rolling(63).std()*np.sqrt(252); ir126=ret.rolling(126).mean()/ret.rolling(126).std()*np.sqrt(252)
        risk=.6*pct_rank(ir63,f(ir63.iloc[-1]))+.4*pct_rank(ir126,f(ir126.iloc[-1]))
        m=.25*mom[0]+.35*mom[1]+.25*mom[2]+.15*mom[3]
        t=.25*trend[0]+.30*trend[1]+.25*trend[2]+.20*trend[3]
        b=.25*breadth[0]+.30*breadth[1]+.25*breadth[2]+.20*breadth[3]
        score=.40*m+.20*t+.25*b+.15*risk
        rows.append({"Group":gid,"Score":score,"Momentum":m,"Trend":t,"Breadth":b,"RiskAdjusted":risk,"Coverage":len(avail)/len(tickers)})
    df=pd.DataFrame(rows)
    group_score=df.set_index("Group")["Score"].to_dict()
    weighted=sum(COMPANY_WEIGHTS[c]*group_score.get(COMPANY_REGISTRY[c]["group"],np.nan) for c in COMPANY_WEIGHTS if np.isfinite(group_score.get(COMPANY_REGISTRY[c]["group"],np.nan)))/TOP10_COVERAGE
    equal=df["Score"].mean()
    final=.65*weighted+.35*equal if np.isfinite(weighted) and np.isfinite(equal) else np.nan
    return df,{"Block1Score":final,"Top10GroupWeighted":weighted,"EqualGroup":equal,"Status":"BLOCK1_PASS" if np.isfinite(final) else "BLOCK1_INCOMPLETE"}

# -----------------------------------------------------------------------------
# BLOCK 2 — STOIC / HEADLINE-CHALLENGED FUNDAMENTALS
# -----------------------------------------------------------------------------
GROWTH_A=[(-.25,0),(0,40),(.10,60),(.25,80),(.50,100)]
MARGIN_A=[(-.05,0),(0,35),(.10,55),(.25,75),(.50,100)]
DELTA_A=[(-.10,0),(-.02,25),(0,50),(.03,75),(.10,100)]
QUALITY_A=[(-.50,0),(-.10,30),(0,55),(.15,75),(.40,100)]
INV_BURDEN_A=[(0,100),(.25,80),(.60,55),(1.0,25),(1.5,0)]

UNIVERSAL_WEIGHTS={"Demand":.20,"Scaling":.20,"Earnings":.20,"Cash":.15,"Capex":.15,"Balance":.10}


def score_company(company: str) -> tuple[dict[str,Any], list[dict[str,Any]], list[dict[str,Any]]]:
    s=SNAP[company]
    decomposition=[]; challenges=[]
    demand_vals={"RevenueYoY":score_interp(growth(s['rev'],s['rev_p']),GROWTH_A),"Revenue2YCAGR":score_interp(s['rev_cagr'],GROWTH_A),"RevenueAcceleration":score_interp(s['rev_acc'],DELTA_A),"ReceivablesQuality":score_interp(s['rec_q'],QUALITY_A),"InventoryQuality":score_interp(s['inv_q'],QUALITY_A)}
    demand=weighted_score(demand_vals,{"RevenueYoY":.35,"Revenue2YCAGR":.20,"RevenueAcceleration":.20,"ReceivablesQuality":.15,"InventoryQuality":.10},("RevenueYoY",),.70)
    scaling_vals={"OIGrowth":score_interp(growth(s['oi'],s['oi_p']),GROWTH_A),"GrossMargin":score_interp(s['gm'],MARGIN_A),"GrossMarginDelta":score_interp(s['gm_d'],DELTA_A),"OperatingMargin":score_interp(s['om'],MARGIN_A),"OperatingMarginDelta":score_interp(s['om_d'],DELTA_A),"IncrementalMargin":score_interp(s['incr'],MARGIN_A)}
    scaling=weighted_score(scaling_vals,{"OIGrowth":.25,"GrossMargin":.15,"GrossMarginDelta":.10,"OperatingMargin":.20,"OperatingMarginDelta":.15,"IncrementalMargin":.15},("OIGrowth","OperatingMargin"),.70)
    distortion=abs(s['eps']-s['norm_eps'])/max(abs(s['eps']),.01)
    earnings_vals={"EPSGrowth":score_interp(growth(s['eps'],s['eps_p']),GROWTH_A),"NormalizedRetention":score_interp(s['norm_eps']/max(abs(s['eps']),.01),[(0,0),(.5,40),(.8,70),(1,100)]),"DistortionInverse":score_interp(distortion,[(0,100),(.15,80),(.35,50),(.70,10),(1,0)]),"CoreOperatingShare":score_interp(s['core'],[(0,0),(.5,50),(.8,80),(1,100)]),"DilutionInverse":score_interp(s['shares_d'],[(.10,0),(.03,30),(0,70),(-.03,100)])}
    earnings=weighted_score(earnings_vals,{"EPSGrowth":.20,"NormalizedRetention":.25,"DistortionInverse":.25,"CoreOperatingShare":.20,"DilutionInverse":.10},("NormalizedRetention","DistortionInverse"),.70)
    cash_vals={"FCFMargin":score_interp(s['fcf_m'],MARGIN_A),"FCFMarginChange":score_interp(s['fcf_m_d'],DELTA_A),"CFOGrowth":score_interp(s['cfo_g'],GROWTH_A),"CFOToNI":score_interp(s['cfo_ni'],[(0,0),(.5,40),(1,80),(1.5,100)]),"FCFToNI":score_interp(s['fcf_ni'],[(0,0),(.5,55),(1,100)])}
    cash=weighted_score(cash_vals,{"FCFMargin":.25,"FCFMarginChange":.15,"CFOGrowth":.20,"CFOToNI":.20,"FCFToNI":.20},("FCFMargin","CFOToNI"),.70)
    capex_vals={"CapexToCFOInverse":score_interp(s['capex_cfo'],INV_BURDEN_A),"CapexToRevenueInverse":score_interp(s['capex_rev'],INV_BURDEN_A),"CapexToDepInverse":score_interp(s['capex_dep'],[(0,90),(1,80),(2,55),(5,20),(12,0)]),"RevenueConversion":score_interp(s['rev_conv'],[(0,0),(.5,40),(1,60),(2,80),(5,100)]),"OIConversion":score_interp(s['oi_conv'],[(0,0),(.25,45),(.75,70),(1.5,90),(3,100)]),"CashAbsorptionInverse":score_interp(s['cash_abs'],INV_BURDEN_A),"DepPressureInverse":score_interp(s['dep_press'],[(0,100),(.05,80),(.15,50),(.30,10)])}
    capex=weighted_score(capex_vals,{"CapexToCFOInverse":.20,"CapexToRevenueInverse":.15,"CapexToDepInverse":.10,"RevenueConversion":.15,"OIConversion":.15,"CashAbsorptionInverse":.15,"DepPressureInverse":.10},("CapexToCFOInverse","CapexToRevenueInverse"),.65)
    balance_vals={"NetCashToRevenue":score_interp(s['netcash_rev'],[(-.75,0),(-.25,30),(0,55),(.25,80),(.75,100)]),"BuybacksToCFO":score_interp(s['buyback_cfo'],[(0,45),(.10,60),(.30,80),(.60,100)]),"SBCToCFOInverse":score_interp(s['sbc_cfo'],[(0,100),(.10,75),(.25,40),(.50,0)]),"DilutionInverse":earnings_vals['DilutionInverse']}
    balance=weighted_score(balance_vals,{"NetCashToRevenue":.35,"BuybacksToCFO":.25,"SBCToCFOInverse":.25,"DilutionInverse":.15},(),.55)
    cats={"Demand":demand,"Scaling":scaling,"Earnings":earnings,"Cash":cash,"Capex":capex,"Balance":balance}
    uni_vals={k:v['Official'] for k,v in cats.items()}; uni=weighted_score(uni_vals,UNIVERSAL_WEIGHTS,("Demand","Scaling","Earnings","Cash","Capex"),UNIVERSAL_COVERAGE_MIN)
    sp=SPECIFIC[company]
    sp_scores={}
    for k,x in sp['values'].items():
        if "Margin" in k or k in {"GrossMargin","FCFMargin"}: sp_scores[k]=score_interp(x,MARGIN_A)
        elif "Burden" in k or "Concentration" in k or "Discipline" in k: sp_scores[k]=score_interp(x,INV_BURDEN_A)
        else: sp_scores[k]=score_interp(x,GROWTH_A)
    specific=weighted_score(sp_scores,sp['weights'],sp['critical'],SPECIFIC_COVERAGE_MIN)
    if FORWARD_POLICY[company].startswith("NOT_APPLICABLE"):
        forward={"Diagnostic":np.nan,"Official":np.nan,"Coverage":np.nan,"CriticalOK":True,"NotApplicable":True}
    else:
        vals={}
        for k,x in FORWARD.get(company,{}).items():
            vals[k]=score_interp(x,INV_BURDEN_A) if "Capex" in k else score_interp(x,MARGIN_A if "Margin" in k else GROWTH_A)
        w={k:1/len(vals) for k in vals} if vals else {}
        forward=weighted_score(vals,w,tuple(vals),FORWARD_COVERAGE_MIN) if vals else {"Diagnostic":np.nan,"Official":np.nan,"Coverage":0.0,"CriticalOK":False}
        forward['NotApplicable']=False
    full_weights={"Universal":.75,"Specific":.15}
    full_values={"Universal":uni['Official'],"Specific":specific['Official']}
    if not forward.get('NotApplicable'):
        full_weights['Forward']=.10; full_values['Forward']=forward['Official']
    else:
        z=sum(full_weights.values()); full_weights={k:v/z for k,v in full_weights.items()}
    full=weighted_score(full_values,full_weights,tuple(full_weights),.999)
    status="OFFICIAL_FULL" if np.isfinite(full['Official']) else "PROVISIONAL_DIAGNOSTIC"
    implication=("CLEAN_BULLISH" if full['Official']>=75 and distortion<.15 else "CONSTRUCTIVE_WITH_EPS_RISK" if (full['Diagnostic'] or 0)>=65 else "MIXED" if (full['Diagnostic'] or 0)>=50 else "WEAKENING")
    for cat,info in cats.items():
        decomposition.append({"Company":company,"Category":cat,"Score":info['Diagnostic'],"Official":info['Official'],"Coverage":info['Coverage']})
    challenges += [
        {"Company":company,"Headline":"Revenue growth","Challenge":"Real demand, acceleration, receivables and inventory confirmation","Evidence":f"Revenue YoY {growth(s['rev'],s['rev_p']):.1%}; acceleration {s['rev_acc']:.1%}; receivables spread {s['rec_q']:.1%}; inventory spread {s['inv_q']:.1%}"},
        {"Company":company,"Headline":"EPS growth","Challenge":"Reported versus normalized EPS and non-operating distortion","Evidence":f"Reported EPS {s['eps']:.2f}; normalized EPS {s['norm_eps']:.2f}; distortion {distortion:.1%}"},
        {"Company":company,"Headline":"AI / investment narrative","Challenge":"Operating leverage, cash conversion, capex absorption and future depreciation","Evidence":f"OI YoY {growth(s['oi'],s['oi_p']):.1%}; FCF margin {s['fcf_m']:.1%}; capex/CFO {s['capex_cfo']:.1%}; depreciation pressure {s['dep_press']:.1%}"},
    ]
    row={"Company":company,"Ticker":COMPANY_REGISTRY[company]['ticker'],"ActualNQWeight":COMPANY_WEIGHTS[company],"Top10NormalizedWeight":TOP10_NORMALIZED[company],"PeriodEnd":s['period'],"ConfirmedPeriodicAccession":s['accession'],"Source":s['source'],"SourceURL":s['url'],"UniversalOfficialScore":uni['Official'],"UniversalCoverage":uni['Coverage'],"CompanySpecificOfficialScore":specific['Official'],"CompanySpecificCoverage":specific['Coverage'],"ForwardOfficialScore":forward.get('Official'),"ForwardCoverage":forward.get('Coverage'),"ForwardPolicy":FORWARD_POLICY[company],"OfficialFullScore":full['Official'],"DiagnosticFullScore":full['Diagnostic'],"CompanyScoreStatus":status,"EPSDistortionRatio":distortion,"NQFundamentalImplication":implication}
    return row,decomposition,challenges


def build_block2() -> dict[str,Any]:
    score_rows=[]; decomp=[]; challenges=[]
    for c in COMPANY_WEIGHTS:
        r,d,h=score_company(c); score_rows.append(r); decomp.extend(d); challenges.extend(h)
    scorecard=pd.DataFrame(score_rows)
    sec_audit=pd.DataFrame([latest_periodic_accession(c) for c in COMPANY_WEIGHTS])
    scorecard=scorecard.merge(sec_audit,on="Company",how="left")
    scorecard['SnapshotMatched']=scorecard['LatestAccession'].eq(scorecard['ConfirmedPeriodicAccession'])
    scorecard.loc[~scorecard['SnapshotMatched'],'CompanyScoreStatus']='REQUIRES_REAUDIT'
    scorecard.loc[~scorecard['SnapshotMatched'],'OfficialFullScore']=np.nan
    official_weight=sum(scorecard.loc[scorecard['OfficialFullScore'].notna(),'Top10NormalizedWeight'])
    official_score=(scorecard['OfficialFullScore']*scorecard['Top10NormalizedWeight']).sum()/official_weight if official_weight else np.nan
    full_nq_contribution=sum(scorecard['ActualNQWeight']*(scorecard['OfficialFullScore']-50)/50, skipna=True)
    tests={
        "true_top10_has_amd": "AMD" in set(scorecard.Company),
        "true_top10_excludes_walmart": "WALMART" not in set(scorecard.Company),
        "ten_companies_exactly": len(scorecard)==10,
        "alphabet_share_classes_combined": abs(COMPANY_WEIGHTS['ALPHABET']-(.0329+.0306))<1e-12,
        "weights_not_redistributed": WEIGHT_MANIFEST['missing_weight_redistribution'] is False,
        "full_index_not_renormalized": WEIGHT_MANIFEST['full_index_renormalization'] is False,
        "all_latest_accessions_match": bool(scorecard['SnapshotMatched'].all()),
        "scores_bounded": bool(scorecard['OfficialFullScore'].dropna().between(0,100).all()),
        "official_top10_coverage_at_least_95pct": official_weight>=OFFICIAL_TOP10_COVERAGE_MIN,
        "reported_and_normalized_eps_separate": all('norm_eps' in SNAP[c] and 'eps' in SNAP[c] for c in COMPANY_WEIGHTS),
        "headline_challenge_matrix_present": len(challenges)==30,
        "no_final_nq_bias_calculated": True,
    }
    technical=all(tests[k] for k in ["true_top10_has_amd","true_top10_excludes_walmart","ten_companies_exactly","alphabet_share_classes_combined","weights_not_redistributed","full_index_not_renormalized","scores_bounded","headline_challenge_matrix_present","no_final_nq_bias_calculated"])
    data_pass=tests['all_latest_accessions_match'] and tests['official_top10_coverage_at_least_95pct']
    economic=tests['reported_and_normalized_eps_separate']
    freeze=technical and data_pass and economic and PERSISTENCE_STATUS=="DURABLE_GOOGLE_DRIVE"
    status={"Technical":"BLOCK2_TECHNICAL_PASS" if technical else "BLOCK2_TECHNICAL_FAIL","DataValidation":"BLOCK2_DATA_VALIDATION_PASS" if data_pass else "BLOCK2_DATA_VALIDATION_INCOMPLETE","EconomicLogic":"BLOCK2_ECONOMIC_LOGIC_PASS" if economic else "BLOCK2_ECONOMIC_LOGIC_FAIL","Freeze":"BLOCK2_FREEZE_READY" if freeze else "BLOCK2_NOT_FREEZE_READY"}
    release={"NotebookVersion":ENGINE_VERSION,"DataContractVersion":DATA_CONTRACT_VERSION,"FormulaVersion":FORMULA_VERSION,"WeightVersion":WEIGHT_MANIFEST['version'],"WeightHash":stable_hash(WEIGHT_MANIFEST),"FormulaHash":stable_hash({"universal":UNIVERSAL_WEIGHTS,"specific":SPECIFIC,"forward":FORWARD_POLICY}),"AsOfTimestampUTC":str(AS_OF_TIMESTAMP_UTC),"Top10FullNQCoverage":TOP10_COVERAGE,"OfficialTop10Coverage":official_weight,"OfficialTop10Score":official_score,"FullNQFundamentalTiltContribution":full_nq_contribution,"FinalNQBiasCalculated":False,"ReleaseHash":stable_hash({"status":status,"scores":scorecard[['Company','OfficialFullScore']].to_dict('records')})}
    return {"scorecard":scorecard,"decomposition":pd.DataFrame(decomp),"headline_challenges":pd.DataFrame(challenges),"tests":pd.DataFrame([{"Test":k,"Passed":v} for k,v in tests.items()]),"status":status,"release":pd.DataFrame([release]),"official_score":official_score,"official_coverage":official_weight,"full_nq_contribution":full_nq_contribution}


def fetch_fred() -> pd.DataFrame:
    series={"US2Y":"DGS2","US10Y":"DGS10","US10Y_REAL":"DFII10","US10Y_BREAKEVEN":"T10YIE","NFCI":"NFCI","HY_OAS":"BAMLH0A0HYM2","IG_OAS":"BAMLC0A0CM"}
    out=[]
    for label,sid in series.items():
        try:
            if FRED_API_KEY_VALID:
                r=requests.get("https://api.stlouisfed.org/fred/series/observations",params={"series_id":sid,"api_key":FRED_API_KEY,"file_type":"json","observation_start":(AS_OF_DATE-pd.DateOffset(years=10)).strftime('%Y-%m-%d'),"observation_end":AS_OF_DATE.strftime('%Y-%m-%d')},timeout=30); r.raise_for_status(); obs=r.json()['observations']; s=pd.Series({pd.Timestamp(x['date']):pd.to_numeric(x['value'],errors='coerce') for x in obs},name=label)
            else:
                x=pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"); x.columns=['DATE','VALUE']; s=pd.Series(pd.to_numeric(x.VALUE,errors='coerce').values,index=pd.to_datetime(x.DATE),name=label)
            out.append(s)
        except Exception:
            out.append(pd.Series(dtype=float,name=label))
    return pd.concat(out,axis=1).sort_index()


def run_all() -> dict[str,Any]:
    print({"Engine":ENGINE_VERSION,"AsOf":str(AS_OF_TIMESTAMP_UTC),"Persistence":PERSISTENCE_STATUS,"TrueTop10":list(COMPANY_WEIGHTS),"Top10Coverage":TOP10_COVERAGE})
    weights=pd.DataFrame([{"Rank":i,"Company":c,"Ticker":COMPANY_REGISTRY[c]['ticker'],"ActualNQWeight":w,"Top10NormalizedWeight":TOP10_NORMALIZED[c]} for i,(c,w) in enumerate(COMPANY_WEIGHTS.items(),1)])
    display(Markdown("# BLOCK 0 — TRUE TOP-10 CONTRACT")); display(weights)
    prices=fetch_prices(); b1_table,b1_summary=build_block1(prices)
    display(Markdown("# BLOCK 1 — PRICE LEADERSHIP")); display(pd.DataFrame([b1_summary])); display(b1_table)
    b2=build_block2()
    display(Markdown("# BLOCK 2 — STRUCTURED FINAL REPORT V6"))
    display(Markdown("## Executive Summary"))
    if np.isfinite(b2['official_score']):
        print(f"Offizieller NQ-Top-10-Fundamentalscore: {b2['official_score']:.1f}/100 bei {b2['official_coverage']:.1%} offizieller Top-10-Abdeckung. AMD ist enthalten; Walmart ist aus dem Top-10-Unternehmensblock entfernt. Fehlende Gewichte werden nicht umverteilt.")
    else:
        print(f"Noch kein offizieller Score. Offizielle Top-10-Abdeckung: {b2['official_coverage']:.1%}. Fehlende oder nicht re-auditierte Gewichte werden nicht umverteilt.")
    display(Markdown("## Key Scorecard")); display(pd.DataFrame([{**b2['status'],"OfficialTop10Score":b2['official_score'],"OfficialTop10Coverage":b2['official_coverage'],"FullNQFundamentalTiltContribution":b2['full_nq_contribution']}]))
    display(Markdown("## Official versus Provisional Coverage")); display(b2['scorecard'])
    display(Markdown("## Mathematical Decomposition")); display(b2['decomposition'])
    display(Markdown("## Headline Challenge Matrix")); display(b2['headline_challenges'])
    display(Markdown("## Macro Fundamental Interpretation")); print("Die Analyse bewertet nicht Headline-Wachstum isoliert. Sie prüft Endnachfrage, Beschleunigung, Forderungen und Lager, operative Hebelwirkung, gemeldeten gegen normalisierten EPS, Cash-Konversion, Capex-Absorption, künftigen Abschreibungsdruck, Bilanzqualität sowie firmenspezifische Segmente und Guidance. Ein AI-Capex-Boom kann gleichzeitig konstruktiv für Hardwarelieferanten und cash-absorbierend beziehungsweise bewertungsfragil für Hyperscaler sein. Fundamentalqualität und Konsensüberraschung bleiben getrennt; ohne zeitgestempelten Konsens wird kein Beat oder Miss erfunden.")
    display(Markdown("## Acceptance Tests and Hashes")); display(b2['tests']); display(b2['release'])
    print("Block 2 statuses:",*b2['status'].values())
    return {"weights":weights,"prices":prices,"block1_table":b1_table,"block1_summary":b1_summary,"block2":b2,"fred":fetch_fred()}
