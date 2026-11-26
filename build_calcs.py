"""Compute every number in the case and write the calculations workbook.

Single source of truth: the design note quotes the numbers this script prints.
Run:  python build_calcs.py
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

OUT = Path(__file__).resolve().parent / "calculations.xlsx"

GB = 1e9
KiB = 1024
CARD_GB = 48.0
RUNTIME_RESERVE = 0.10
BW = 960.0  # GB/s, RTX 6000 Ada
FP16_DENSE = 91.0  # TFLOPS
PREFILL_EFF = 0.45
DECODE_EFF = 0.70

MODELS = {
    # name: (params_B, layers, kv_heads, head_dim, w4, w8, decode_measured)
    "S": (8, 32, 8, 128, 5.2, 8.9, 118),
    "M": (14, 40, 8, 128, 8.9, 15.4, 74),
    "L": (32, 64, 8, 128, 19.0, 34.5, 36),
    "XL": (70, 80, 8, 128, 40.5, 73.0, 15),
}

TURNS = 6
PROMPT = 3200
OUTPUT = 280
SEASON_DAYS = 120
NORMAL_DAYS = 245


def kv_bytes(layers: int, heads: int, dim: int, bits: int) -> float:
    return 2 * layers * heads * dim * (bits / 8)


def capacity() -> list[dict]:
    rows = []
    usable = CARD_GB * (1 - RUNTIME_RESERVE)
    for name, (p, layers, heads, dim, w4, w8, dec) in MODELS.items():
        for bits, w in ((4, w4), (8, w8)):
            if bits == 8 and name in {"M", "S", "XL"} and name != "XL":
                continue  # only L and XL are asked for; add others on demand
            kv_tok = kv_bytes(layers, heads, dim, 16)
            pool = usable - w
            seq_fp16 = kv_tok * 3500 / GB
            seq_int8 = seq_fp16 / 2
            rows.append(
                {
                    "model": name,
                    "params_B": p,
                    "weights_bits": bits,
                    "weights_gb": w,
                    "kv_kib_fp16": kv_tok / KiB,
                    "kv_kib_int8": kv_tok / KiB / 2,
                    "kv_pool_gb": round(pool, 2),
                    "seq_fp16": int(pool // seq_fp16) if pool > 0 else 0,
                    "seq_int8": int(pool // seq_int8) if pool > 0 else 0,
                    "seq_gb_fp16": round(seq_fp16, 3),
                }
            )
    return rows


def throughput() -> list[dict]:
    usable = CARD_GB * (1 - RUNTIME_RESERVE)
    out = []
    for name, (p, layers, heads, dim, w4, w8, dec) in MODELS.items():
        kv_seq = kv_bytes(layers, heads, dim, 16) * 3500 / GB
        pool = usable - w4
        seqs = max(int(pool // kv_seq), 1)
        step_bytes = w4 + seqs * kv_seq  # weights + all resident KV
        step_ms = step_bytes / BW * 1000
        agg_ideal = seqs / (step_ms / 1000)
        agg_real = agg_ideal * DECODE_EFF
        prefill = (FP16_DENSE * PREFILL_EFF) / (2 * p * 1e9) * 1e12 / 1e12
        prefill = (FP16_DENSE * 1e12 * PREFILL_EFF) / (2 * p * 1e9)
        out.append(
            {
                "model": name,
                "weights_gb": w4,
                "seqs": seqs,
                "step_ms": round(step_ms, 1),
                "decode_ideal": round(agg_ideal),
                "decode_real": round(agg_real),
                "decode_single": dec,
                "prefill_tok_s": round(prefill),
            }
        )
    return out


def traffic() -> list[dict]:
    scen = [
        ("Normal day", 1500),
        ("Counselling season", 22000),
        ("Result day", 60000),
    ]
    rows = []
    for label, chats in scen:
        turns = chats * TURNS
        peak_turns_h = turns * 0.70 / 4
        rows.append(
            {
                "scenario": label,
                "chats_day": chats,
                "turns_day": turns,
                "turns_s_peak": round(peak_turns_h / 3600, 2),
                "prompt_tok_s_raw": round(peak_turns_h / 3600 * PROMPT),
                "output_tok_s": round(peak_turns_h / 3600 * OUTPUT),
                "prompt_tok_s_design": round(peak_turns_h / 3600 * 880),
            }
        )
    return rows


def batch_plan() -> dict:
    backlog = 240_000
    jds = 3_500
    embed_tok = 1900
    jd_tok = 700
    # embedding pass, 0.3B encoder
    embed_flops = 2 * 0.3e9 * backlog * embed_tok
    embed_h = embed_flops / (FP16_DENSE * 1e12 * PREFILL_EFF) / 3600
    # cross-encoder re-rank: top-50 per JD
    rerank_pairs = jds * 50
    rerank_flops = 2 * 0.3e9 * rerank_pairs * (embed_tok + jd_tok)
    rerank_h = rerank_flops / (FP16_DENSE * 1e12 * PREFILL_EFF) / 3600
    # LLM explanation, top-10 per JD on Model S
    calls = jds * 10
    s_prefill = (FP16_DENSE * 1e12 * PREFILL_EFF) / (2 * 8e9)
    explain_prefill_h = calls * 2600 / s_prefill / 3600
    explain_out_h = calls * 120 / 1276 / 3600
    ocr_h = 72_000 * 8 / 3600 / 6  # 6-way CPU parallelism for the 30% scanned
    total = embed_h + rerank_h + explain_prefill_h + explain_out_h + ocr_h
    weekly = total * 4000 / backlog
    return {
        "embed_h": round(embed_h, 1),
        "rerank_h": round(rerank_h, 1),
        "explain_h": round(explain_prefill_h + explain_out_h, 1),
        "ocr_h": round(ocr_h, 1),
        "total_h": round(total, 1),
        "weekly_h": round(weekly, 2),
        "nights_at_8h": round(total / 8, 1),
    }


def tco() -> dict:
    power_kwh_normal = 8 * 1.2 + 16 * 0.35
    power_kwh_season = 10 * 1.2 + 14 * 0.35
    tariff = 9.80
    power_year = (NORMAL_DAYS * power_kwh_normal + SEASON_DAYS * power_kwh_season) * tariff
    ups_ac = 210_000
    ups_upkeep = 6500 * 12
    ws = 1_480_000
    gpu2 = 890_000
    eng = 1_800_000
    junior = 55_000 * 12
    amc = 0.05 * (ws + ups_ac)
    spare_nvme = 40_000

    years = 3
    junior_months = 32
    people = eng * years + junior * junior_months / 12
    power3 = power_year * years
    ups3 = ups_upkeep * years
    amc3 = amc * years
    one_card = ws + ups_ac + spare_nvme + people + power3 + ups3 + amc3
    two_card = one_card + gpu2 + (gpu2 * 0.05) * 2.5 + power_year * 0.35

    chats = {"y1": 1_400_000, "y2": 2_900_000, "y3": 3_100_000}
    total_chats = sum(chats.values())
    tokens_in = TURNS * PROMPT
    tokens_out = TURNS * OUTPUT
    cloud_per_chat = tokens_in / 1e6 * 42 + tokens_out / 1e6 * 170
    cloud3 = total_chats * cloud_per_chat
    resumes3 = 240_000 + 4000 * 156
    cloud_per_resume = 2600 / 1e6 * 42 + 120 / 1e6 * 170
    return {
        "power_kwh_normal": power_kwh_normal,
        "power_kwh_season": power_kwh_season,
        "power_year": round(power_year),
        "people3": round(people),
        "one_card": round(one_card),
        "two_card": round(two_card),
        "cash_excl_sunk": round(one_card - ws - ups_ac),
        "chats3": total_chats,
        "cost_per_chat": round(one_card / total_chats, 2),
        "cost_per_1000_chats": round(one_card / total_chats * 1000),
        "marginal_per_1000": round((power3 + ups3 + amc3 + spare_nvme) / total_chats * 1000),
        "people_share": round(people / one_card * 100),
        "cloud_per_chat": round(cloud_per_chat, 3),
        "cloud3": round(cloud3),
        "breakeven_chats_year": round(one_card / cloud_per_chat / 3),
        "resumes3": resumes3,
        "cloud_per_resume": round(cloud_per_resume, 3),
        "onprem_per_resume": round((people * 0.15) / resumes3 + 0.06, 2),
        "cloud_resumes3": round(resumes3 * cloud_per_resume),
    }


def rental() -> dict:
    hours_year = 24 * 365
    busy = 8.5 * 365
    spare = hours_year - busy
    sellable = spare * 0.80
    price = 60
    util = 0.40
    revenue = sellable * util * price
    return {
        "spare_h": round(spare),
        "sellable_h": round(sellable),
        "price": price,
        "revenue_year": round(revenue),
    }


def style(ws, header: list[str]) -> None:
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F3864")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "A2"


def write_workbook(c: list[dict], t: list[dict], tr: list[dict], b: dict, m: dict, r: dict) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "1 KV cache and concurrency"
    ws.append(["B1. KV cache per token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_element"])
    ws.append(["Card 48 GB; runtime reserve 10% of the card = 4.8 GB; usable 43.2 GB; sequence length 3500 tokens."])
    ws.append([])
    style(ws, ["Model", "Params (B)", "Weights (bits)", "Weights (GB)", "KV/token fp16 (KiB)",
               "KV/token int8 (KiB)", "KV pool (GB)", "Seqs fp16", "Seqs int8", "GB per 3500-tok seq"])
    for row in c:
        ws.append([row["model"], row["params_B"], row["weights_bits"], row["weights_gb"],
                   round(row["kv_kib_fp16"], 1), round(row["kv_kib_int8"], 1), row["kv_pool_gb"],
                   row["seq_fp16"], row["seq_int8"], row["seq_gb_fp16"]])
    for col, w in zip("ABCDEFGHIJ", [10, 12, 14, 12, 16, 16, 12, 10, 10, 16]):
        ws.column_dimensions[col].width = w

    ws2 = wb.create_sheet("2 Throughput")
    ws2.append(["B2. Roofline at 960 GB/s and 91 TFLOPS fp16 dense; prefill efficiency 45%, decode efficiency 70%."])
    ws2.append(["Consistency check: measured single-stream decode / (BW / weights) lands at 63-71% for all four models."])
    ws2.append([])
    style(ws2, ["Model", "Weights 4-bit (GB)", "Concurrent seqs", "Decode step (ms)",
                "Aggregate decode ideal (tok/s)", "Aggregate decode real (tok/s)",
                "Measured single stream (tok/s)", "Prefill (tok/s)"])
    for row in t:
        ws2.append([row["model"], row["weights_gb"], row["seqs"], row["step_ms"],
                    row["decode_ideal"], row["decode_real"], row["decode_single"], row["prefill_tok_s"]])
    for col, w in zip("ABCDEFGH", [10, 18, 16, 16, 28, 26, 26, 16]):
        ws2.column_dimensions[col].width = w

    ws3 = wb.create_sheet("3 Worst hour demand")
    ws3.append(["B2. 70% of the day's traffic in 4 hours (7-11 PM); 6 turns per chat."])
    ws3.append(["'design' prompt tokens assume a 2,200-token context cap with 60% prefix-cache hit = ~880 new tokens/turn."])
    ws3.append([])
    style(ws3, ["Scenario", "Chats/day", "Turns/day", "Turns/s in peak hour",
                "Prompt tok/s (3,200 raw)", "Output tok/s", "Prompt tok/s (design)"])
    for row in tr:
        ws3.append([row["scenario"], row["chats_day"], row["turns_day"], row["turns_s_peak"],
                    row["prompt_tok_s_raw"], row["output_tok_s"], row["prompt_tok_s_design"]])
    for col, w in zip("ABCDEFG", [22, 12, 12, 20, 24, 14, 22]):
        ws3.column_dimensions[col].width = w

    ws4 = wb.create_sheet("4 Batch plan")
    ws4.append(["B3. Hirrx full backlog of 240,000 resumes against 3,500 JDs, overnight window 23:00-07:00."])
    ws4.append([])
    style(ws4, ["Stage", "GPU-hours"])
    for k, label in [
        ("embed_h", "Embeddings: 240k resumes x 1,900 tokens, 0.3B encoder"),
        ("rerank_h", "Cross-encoder re-rank: 3,500 JDs x top-50"),
        ("explain_h", "LLM explanations: 3,500 JDs x top-10 on Model S"),
        ("ocr_h", "OCR of 72,000 scanned PDFs (CPU, 6-way)"),
        ("total_h", "TOTAL per full backlog"),
        ("weekly_h", "Weekly increment (4,000 new resumes)"),
        ("nights_at_8h", "Nights needed at 8 h/night"),
    ]:
        ws4.append([label, b[k]])
    ws4.column_dimensions["A"].width = 62

    ws5 = wb.create_sheet("5 Three-year TCO")
    ws5.append(["F1. Three years, Dec 2026 - Nov 2029. Engineer CTC assumed INR 18,00,000/yr; junior INR 55,000/month from Apr 2027."])
    ws5.append([])
    style(ws5, ["Line", "One GPU (INR)", "Two GPUs (INR)"])
    ws5.append(["Workstation (sunk)", 1_480_000, 1_480_000])
    ws5.append(["UPS and cooling", 210_000, 210_000])
    ws5.append(["Second GPU", 0, 890_000])
    ws5.append(["People (3 years)", m["people3"], m["people3"]])
    ws5.append(["Power (3 years)", round(m["power_year"] * 3), round(m["power_year"] * 3.35)])
    ws5.append(["UPS/AC upkeep (3 years)", 6500 * 12 * 3, 6500 * 12 * 3])
    ws5.append(["AMC and spares (3 years)", round(0.05 * (1_480_000 + 210_000) * 3 + 40_000),
                round(0.05 * (1_480_000 + 210_000 + 890_000) * 3 + 40_000)])
    ws5.append(["TOTAL", m["one_card"], m["two_card"]])
    ws5.append(["Cash excluding the sunk workstation", m["cash_excl_sunk"], m["two_card"] - 1_480_000])
    ws5.append([])
    ws5.append(["Chats over three years", m["chats3"]])
    ws5.append(["All-in cost per chat (INR)", m["cost_per_chat"]])
    ws5.append(["All-in cost per 1,000 chats (INR)", m["cost_per_1000_chats"]])
    ws5.append(["Marginal cost per 1,000 chats (power + AMC only)", m["marginal_per_1000"]])
    ws5.append(["People as a share of TCO (%)", m["people_share"]])
    ws5.append(["Cloud cost per chat (INR)", m["cloud_per_chat"]])
    ws5.append(["Cloud, three years (INR)", m["cloud3"]])
    ws5.append(["Break-even volume (chats/year)", m["breakeven_chats_year"]])
    ws5.append([])
    ws5.append(["Resumes ranked over three years", m["resumes3"]])
    ws5.append(["On-prem cost per resume (INR)", m["onprem_per_resume"]])
    ws5.append(["Cloud cost per resume (INR)", m["cloud_per_resume"]])
    ws5.append(["Cloud, resumes over three years (INR)", m["cloud_resumes3"]])
    ws5.column_dimensions["A"].width = 52
    ws5.column_dimensions["B"].width = 18
    ws5.column_dimensions["C"].width = 18

    ws6 = wb.create_sheet("6 Rental")
    ws6.append(["F2. Renting spare GPU hours at night."])
    ws6.append([])
    style(ws6, ["Item", "Value"])
    ws6.append(["Hours in a year", 24 * 365])
    ws6.append(["Busy hours (8.5 h/day average)", round(8.5 * 365)])
    ws6.append(["Spare hours", r["spare_h"]])
    ws6.append(["Sellable at 80% (maintenance reserve)", r["sellable_h"]])
    ws6.append(["Price per GPU-hour (INR)", r["price"]])
    ws6.append(["Utilisation assumed", 0.40])
    ws6.append(["Revenue per year (INR)", r["revenue_year"]])
    ws6.append(["As a share of three-year TCO (%)", round(r["revenue_year"] * 3 / m["one_card"] * 100, 2)])
    ws6.column_dimensions["A"].width = 44
    ws6.column_dimensions["B"].width = 16

    wb.save(OUT)


def main() -> None:
    c = capacity()
    t = throughput()
    tr = traffic()
    b = batch_plan()
    m = tco()
    r = rental()
    write_workbook(c, t, tr, b, m, r)
    print("=== B1 KV cache and concurrency ===")
    for row in c:
        print(row)
    print("\n=== B2 roofline ===")
    for row in t:
        print(row)
    print("\n=== B2 worst hour ===")
    for row in tr:
        print(row)
    print("\n=== B3 batch ===")
    print(b)
    print("\n=== F1 TCO ===")
    for k, v in m.items():
        print(f"  {k}: {v}")
    print("\n=== F2 rental ===")
    print(r)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
