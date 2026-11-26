"""Build the synthetic dataset that ships with the prototype.

Everything in here is invented for the assignment. No real college, fee,
counselling rule or person is represented; the figures follow the shape of an
Indian MBBS fee notice so that the retrieval and numeric-guard problems are the
real ones (two sessions, table rows, a nursing-fee distractor, a state bond that
differs from its neighbour).

Run:  python tools/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COLLEGES = [
    # code, name, city, state, authority
    ("SMC-PAT", "Shanti Medical College", "Patna", "Bihar", "Bihar Private Medical Colleges Fee Committee"),
    ("GIMS-VNS", "Ganges Institute of Medical Sciences", "Varanasi", "Uttar Pradesh", "UP Medical Education Fee Board"),
    ("DMC-HYD", "Deccan Medical College", "Hyderabad", "Telangana", "Telangana Fee Regulatory Committee"),
    ("RMCH-KOL", "Rajarhat Medical College & Hospital", "Kolkata", "West Bengal", "West Bengal Fee Regulatory Committee"),
    ("KIMS-BBS", "Kalinga Institute of Medical Sciences", "Bhubaneswar", "Odisha", "Odisha Medical Education Fee Panel"),
    ("SMC-ROH", "Saraswati Medical College", "Rohtak", "Haryana", "Haryana Medical Education Fee Panel"),
]

# tuition, hostel, deposit, other, seats   (per year, INR, case data)
FEES = {
    "SMC-ROH": {"2526": (890000, 105000, 200000, 36000, 100), "2627": (955000, 112000, 210000, 38000, 100)},
    "SMC-PAT": {"2526": (950000, 120000, 250000, 45000, 150), "2627": (1040000, 130000, 250000, 48000, 150)},
    "GIMS-VNS": {"2526": (1100000, 95000, 200000, 38000, 100), "2627": (1185000, 105000, 200000, 41000, 100)},
    "DMC-HYD": {"2526": (1375000, 140000, 300000, 55000, 150), "2627": (1460000, 148000, 300000, 58000, 150)},
    "RMCH-KOL": {"2526": (680000, 84000, 150000, 32000, 125), "2627": (725000, 92000, 150000, 34000, 125)},
    "KIMS-BBS": {"2526": (1025000, 110000, 200000, 41000, 100), "2627": (1095000, 118000, 200000, 44000, 100)},
}


def inr(v: int) -> str:
    s = str(v)
    if len(s) <= 3:
        return "INR " + s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "INR " + ",".join(parts + [tail])


def fee_doc(code: str, session: str) -> dict:
    name, city, state, authority = next(c for c in COLLEGES if c[0] == code)[1:5]
    tuition, hostel, deposit, other, seats = FEES[code][session.replace("-", "")]
    total = tuition + hostel + deposit + other
    label = "2025-26" if session == "2526" else "2026-27"
    prev = "2024-25" if session == "2526" else "2025-26"
    text = f"""Official MBBS fee notice for {name}, {city}, {state}.
Session {label}. Issued by the {authority}. This notice replaces the {prev} notice for the same programme and college.

| Head | Amount per year |
| Annual tuition fee | {inr(tuition)} |
| Hostel and mess charges | {inr(hostel)} |
| One-time admission and caution deposit | {inr(deposit)} |
| Other charges, library, laboratory and insurance | {inr(other)} |
| Total payable in the first year | {inr(total)} |

Seats approved for MBBS in this session: {seats}.
The tuition fee above is for the MBBS programme only. Fees for BSc Nursing, BDS and postgraduate courses are notified separately and are not covered by this notice.
"""
    return {
        "doc_id": f"DOC-{code}-FEE-{session}",
        "title": f"{name}, {city} — MBBS fee notice {label}",
        "session": label,
        "state": state,
        "college": name,
        "doc_type": "fee_notice",
        "status": "superseded" if session == "2526" else "active",
        "effective_date": "2025-05-20" if session == "2526" else "2026-05-18",
        "source": f"https://fee.{state.lower().replace(' ', '')}.gov.in/notices/{code.lower()}-{session}.pdf",
        "text": text.strip(),
    }


BONDS = [
    ("DOC-BOND-BR-2627", "Bihar", "10,00,000", "3 years", "a designated district hospital in Bihar"),
    ("DOC-BOND-UP-2627", "Uttar Pradesh (UP)", "5,00,000", "5 years", "a primary health centre in a rural district of Uttar Pradesh"),
    ("DOC-BOND-HR-2627", "Haryana", "10,00,000", "2 years", "a designated rural or district hospital in Haryana"),
    ("DOC-BOND-WB-2627", "West Bengal", "8,00,000", "1 year", "a rural health centre in West Bengal"),
]


def bond_doc(doc_id: str, state: str, amount: str, period: str, place: str) -> dict:
    text = f"""State service bond rule for MBBS graduates admitted against state quota seats.
State: {state}. Session 2026-27. This rule is issued by the state medical education department.

| Item | Rule |
| Bond amount | INR {amount} |
| Service period | {period} in {place} |
| When the bond is invoked | If the candidate does not join the allotted college, or leaves the course before completion |
| Waiver | Waived only for candidates who serve in a designated aspirational district for the full bond period |

This bond applies only to state quota MBBS seats in {state}. It does not apply to all-India quota seats, to BDS seats, or to postgraduate seats.
"""
    return {
        "doc_id": doc_id,
        "title": f"{state} MBBS state quota bond rule 2026-27",
        "session": "2026-27",
        "state": state.split(" (")[0],
        "college": None,
        "doc_type": "bond_rule",
        "status": "active",
        "effective_date": "2026-04-30",
        "source": f"https://meded.{state.split()[0].lower()}.gov.in/bond-2026-27.pdf",
        "text": text.strip(),
    }


def build_documents() -> list[dict]:
    docs: list[dict] = []
    for code, *_ in COLLEGES:
        docs.append(fee_doc(code, "2526"))
        docs.append(fee_doc(code, "2627"))
    for b in BONDS:
        docs.append(bond_doc(*b))

    docs.append(
        {
            "doc_id": "DOC-NMC-SEAT-2627",
            "title": "NMC state-wise MBBS seat matrix 2026-27",
            "session": "2026-27",
            "state": None,
            "college": None,
            "doc_type": "seat_matrix",
            "status": "active",
            "effective_date": "2026-06-01",
            "source": "https://nmc.org.in/seat-matrix-2026-27.pdf",
            "text": """National Medical Commission approved MBBS seat matrix for session 2026-27.

| State | Government seats | Private seats | Total MBBS seats |
| Bihar | 1,250 | 1,150 | 2,400 |
| Uttar Pradesh | 3,900 | 3,100 | 7,000 |
| Telangana | 2,850 | 3,650 | 6,500 |
| West Bengal | 1,900 | 1,250 | 3,150 |
| Odisha | 1,100 | 850 | 1,950 |
| Haryana | 1,150 | 1,050 | 2,200 |

The matrix above is the approved intake for the 2026-27 session and supersedes the 2025-26 matrix in full.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-NMC-SEAT-2526",
            "title": "NMC state-wise MBBS seat matrix 2025-26",
            "session": "2025-26",
            "state": None,
            "college": None,
            "doc_type": "seat_matrix",
            "status": "superseded",
            "effective_date": "2025-06-01",
            "source": "https://nmc.org.in/seat-matrix-2025-26.pdf",
            "text": """National Medical Commission approved MBBS seat matrix for session 2025-26.

| State | Government seats | Private seats | Total MBBS seats |
| Bihar | 1,200 | 1,050 | 2,250 |
| Uttar Pradesh | 3,800 | 3,000 | 6,800 |
| Telangana | 2,800 | 3,500 | 6,300 |
| West Bengal | 1,850 | 1,200 | 3,050 |
| Odisha | 1,050 | 800 | 1,850 |
| Haryana | 1,100 | 1,000 | 2,100 |

This matrix is withdrawn for the 2026-27 session and must not be quoted for current admissions.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-MCC-RULES-2627",
            "title": "MCC all-India counselling rules 2026-27",
            "session": "2026-27",
            "state": None,
            "college": None,
            "doc_type": "counselling_rules",
            "status": "active",
            "effective_date": "2026-06-10",
            "source": "https://mcc.nic.in/rules-2026-27.pdf",
            "text": """Medical Counselling Committee rules for all-India quota counselling, session 2026-27.

| Rule | Detail |
| Number of rounds | Four rounds, followed by a stray vacancy round |
| Last date to report to the allotted college | 20 September 2026 for the final round |
| Forfeiture if a candidate resigns a seat in round 2 | The tuition fee for the first year is forfeited, and the candidate is barred from the stray vacancy round |
| Forfeiture if a candidate does not join after round 1 | Only the caution deposit is forfeited |
| Seat upgrade | A candidate who accepts an upgraded seat in round 3 loses the earlier seat immediately |

A candidate who resigns after the final reporting date forfeits the full first-year fee. The rule above is applied uniformly to government and private colleges.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-BIHAR-COUNS-2627",
            "title": "Bihar state quota counselling guidelines 2026-27",
            "session": "2026-27",
            "state": "Bihar",
            "college": None,
            "doc_type": "counselling_rules",
            "status": "active",
            "effective_date": "2026-06-15",
            "source": "https://bceceb.bihar.gov.in/guidelines-2026-27.pdf",
            "text": """Bihar Combined Entrance Competitive Examination Board guidelines for state quota counselling 2026-27.

| Item | Detail |
| Eligibility | Domicile certificate of Bihar, or a Bihar school record for the qualifying examination |
| Seat reservation | 10 percent for economically weaker sections, as notified |
| Document verification | Original documents must be produced at the nodal centre in Patna |
| Fee payment | Tuition is paid to the college directly after seat allotment |

State quota seats in Bihar private medical colleges are allotted only through this board. No agent or broker can secure a seat.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-SMC-PAT-PG-2627",
            "title": "Shanti Medical College MD fee notice 2026-27",
            "session": "2026-27",
            "state": "Bihar",
            "college": "Shanti Medical College",
            "doc_type": "fee_notice_pg",
            "status": "active",
            "effective_date": "2026-05-25",
            "source": "https://fee.bihar.gov.in/notices/smc-pat-md-2627.pdf",
            "text": """Official postgraduate fee notice for Shanti Medical College, Patna.
Session 2026-27. This notice covers MD and MS programmes only and does not change the MBBS fee.

| Head | Amount per year |
| Annual tuition fee for MD and MS | INR 18,50,000 |
| Hostel and mess charges | INR 1,45,000 |
| One-time caution deposit | INR 3,00,000 |

Postgraduate seats are filled only through NEET-PG counselling.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-NMC-FACULTY-NORMS",
            "title": "NMC minimum faculty norms for medical colleges",
            "session": "2026-27",
            "state": None,
            "college": None,
            "doc_type": "regulation",
            "status": "active",
            "effective_date": "2026-01-15",
            "source": "https://nmc.org.in/faculty-norms.pdf",
            "text": """NMC minimum faculty norms for undergraduate medical colleges. Issued by the National Medical Commission (NMC).

| Post | Minimum qualification | Minimum teaching experience |
| Professor | MD or MS or DNB in the subject | 8 years, of which 4 years as Associate Professor |
| Associate Professor | MD or MS or DNB in the subject | 5 years, of which 2 years as Assistant Professor |
| Assistant Professor | MD or MS or DNB in the subject | 3 years of teaching experience |
| Senior Resident | MD or MS or DNB in the subject, or MBBS for a non-teaching post | 1 year |
| Tutor | MBBS | None |

Experience is counted from the date of registration with the state medical council. Experience in a non-teaching post is counted at half weight.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-WB-FEE-ORDER-2627",
            "title": "West Bengal fee regulatory order for private medical colleges",
            "session": "2026-27",
            "state": "West Bengal",
            "college": None,
            "doc_type": "fee_order",
            "status": "active",
            "effective_date": "2026-05-05",
            "source": "https://wbfrc.gov.in/order-2026-27.pdf",
            "text": """West Bengal Fee Regulatory Committee order on tuition for private medical colleges, session 2026-27.

| Item | Detail |
| Tuition ceiling for MBBS | INR 8,00,000 per year for private medical colleges in West Bengal |
| Annual increase allowed | Not more than 8 percent over the previous session |
| Colleges above the ceiling | Must refund the excess within 30 days of this order |
| Hostel charges | Regulated separately and cannot be bundled with tuition |

No private medical college in West Bengal may charge tuition above the ceiling in this order.""",
        }
    )

    # Noise / distractors -------------------------------------------------
    docs.append(
        {
            "doc_id": "DOC-NOISE-NURSING-2627",
            "title": "Rajarhat Medical College BSc Nursing fee notice 2026-27",
            "session": "2026-27",
            "state": "West Bengal",
            "college": "Rajarhat Medical College & Hospital",
            "doc_type": "fee_notice_nursing",
            "status": "active",
            "effective_date": "2026-05-22",
            "source": "https://fee.westbengal.gov.in/notices/rmch-nursing-2627.pdf",
            "text": """Official fee notice for the BSc Nursing programme at Rajarhat Medical College & Hospital, Kolkata.
Session 2026-27. This notice covers BSc Nursing only and does not apply to MBBS.

| Head | Amount per year |
| Annual tuition fee for BSc Nursing | INR 1,45,000 |
| Hostel and mess charges | INR 78,000 |
| One-time admission deposit | INR 40,000 |

Nursing seats are filled through the state nursing entrance examination.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-NOISE-TRANSPORT-2627",
            "title": "Saraswati Medical College transport and bus fee circular",
            "session": "2026-27",
            "state": "Haryana",
            "college": "Saraswati Medical College",
            "doc_type": "circular",
            "status": "active",
            "effective_date": "2026-05-28",
            "source": "https://smcrohtak.example.edu.in/circulars/bus-2627.pdf",
            "text": """Transport circular for Saraswati Medical College, Rohtak, session 2026-27.

| Route | Annual bus fee |
| Rohtak city to college | INR 28,000 |
| Jhajjar to college | INR 34,000 |
| Sonipat to college | INR 41,000 |

The bus fee is optional and is not part of the tuition fee. Students who use their own transport do not pay the bus fee.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-NOISE-SCHOLARSHIP-2627",
            "title": "Post-matric scholarship circular for medical students",
            "session": "2026-27",
            "state": None,
            "college": None,
            "doc_type": "circular",
            "status": "active",
            "effective_date": "2026-07-01",
            "source": "https://scholarships.gov.in/circular-2627.pdf",
            "text": """Post-matric scholarship circular for students in recognised medical colleges, session 2026-27.

| Item | Detail |
| Family income ceiling | INR 2,50,000 per year |
| Scholarship amount | INR 60,000 per year towards tuition |
| Application window | 1 July 2026 to 31 October 2026 |
| Mode of payment | Direct benefit transfer to the student account |

The scholarship is credited to the student and does not reduce the fee printed on the college notice.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-NOISE-RESEARCH",
            "title": "CGPRR research brief on health financing",
            "session": "2026-27",
            "state": None,
            "college": None,
            "doc_type": "research",
            "status": "active",
            "effective_date": "2026-08-12",
            "source": "https://cgprr.com/research/health-financing",
            "text": """CGPRR research brief: state health financing and medical education capacity.

| Indicator | Value |
| Public health spend as share of state budget | 4.1 percent |
| Medical colleges added in five years | 92 |
| Districts without a district hospital | 61 |

The brief argues that seat expansion without faculty supply raises the cost of teaching posts. It contains no fee or bond figures for any individual college.""",
        }
    )
    docs.append(
        {
            "doc_id": "DOC-NOISE-TENDER",
            "title": "Wehold Projects hospital construction tender",
            "session": "2026-27",
            "state": "West Bengal",
            "college": None,
            "doc_type": "tender",
            "status": "active",
            "effective_date": "2026-09-02",
            "source": "https://weholdprojects.com/tenders/hospital-2627",
            "text": """Wehold Projects invitation to tender for a 300-bed hospital in New Town, Kolkata.

| Item | Detail |
| Estimate | INR 41,00,00,000 |
| Site | Plot 14, New Town, Kolkata |
| Last date for bids | 30 October 2026 |
| Earnest money | INR 41,00,000 |

Bidders must show three completed hospital projects of at least 200 beds.""",
        }
    )
    return docs


def build_questions() -> list[dict]:
    en = "en"
    hi = "hi-rom"
    bn = "bn-rom"
    q: list[tuple[str, str, bool, list[str], str | None]] = [
        # qid, question, answerable, gold, session
        ("Q01", "What is the annual MBBS tuition fee at Shanti Medical College, Patna for 2026-27?", True, ["DOC-SMC-PAT-FEE-2627"], "2026-27"),
        ("Q02", "What was the annual MBBS tuition fee at Shanti Medical College, Patna in 2025-26?", True, ["DOC-SMC-PAT-FEE-2526"], "2025-26"),
        ("Q03", "What is the MBBS bond amount in Haryana for 2026-27?", True, ["DOC-BOND-HR-2627"], None),
        ("Q04", "What is the MBBS bond amount in Uttar Pradesh?", True, ["DOC-BOND-UP-2627"], None),
        ("Q05", "How many years of service does the Bihar MBBS bond require?", True, ["DOC-BOND-BR-2627"], None),
        ("Q06", "What is the hostel and mess fee at Rajarhat Medical College, Kolkata for 2026-27?", True, ["DOC-RMCH-KOL-FEE-2627"], "2026-27"),
        ("Q07", "How many MBBS seats are approved at Kalinga Institute of Medical Sciences for 2026-27?", True, ["DOC-KIMS-BBS-FEE-2627"], "2026-27"),
        ("Q08", "What is the total payable in the first year at Deccan Medical College, Hyderabad for 2026-27?", True, ["DOC-DMC-HYD-FEE-2627"], "2026-27"),
        ("Q09", "What is the annual tuition fee for MD at Shanti Medical College, Patna?", True, ["DOC-SMC-PAT-PG-2627"], "2026-27"),
        ("Q10", "Which document gives the state-wise MBBS seat matrix for 2026-27?", True, ["DOC-NMC-SEAT-2627"], "2026-27"),
        ("Q11", "What is the forfeiture rule if a candidate resigns a seat in round 2?", True, ["DOC-MCC-RULES-2627"], None),
        ("Q12", "Does the Haryana bond apply to all-India quota seats?", True, ["DOC-BOND-HR-2627"], None),
        ("Q13", "What is the caution deposit at Ganges Institute of Medical Sciences, Varanasi for 2026-27?", True, ["DOC-GIMS-VNS-FEE-2627"], "2026-27"),
        ("Q14", "What is the tuition fee at Saraswati Medical College, Rohtak in 2025-26?", True, ["DOC-SMC-ROH-FEE-2526"], "2025-26"),
        ("Q15", "How much is the West Bengal MBBS bond?", True, ["DOC-BOND-WB-2627"], None),
        ("Q16", "What is the BSc Nursing fee at Rajarhat Medical College?", True, ["DOC-NOISE-NURSING-2627"], "2026-27"),
        ("Q17", "What is the annual bus fee at Saraswati Medical College, Rohtak?", True, ["DOC-NOISE-TRANSPORT-2627"], "2026-27"),
        ("Q18", "What is the minimum teaching experience for an Assistant Professor under NMC norms?", True, ["DOC-NMC-FACULTY-NORMS"], None),
        ("Q19", "What tuition ceiling does the West Bengal fee order set for private medical colleges?", True, ["DOC-WB-FEE-ORDER-2627"], "2026-27"),
        ("Q20", "What is the last date to report to the allotted college in the final round?", True, ["DOC-MCC-RULES-2627"], None),
        ("Q21", "patna me mbbs ki fees kitni hai 2026-27 me", True, ["DOC-SMC-PAT-FEE-2627"], "2026-27"),
        ("Q22", "haryana me bond kitna hai", True, ["DOC-BOND-HR-2627"], None),
        ("Q23", "kolkata medical college me hostel fees kitni hai", True, ["DOC-RMCH-KOL-FEE-2627"], "2026-27"),
        ("Q24", "bihar bond me kitne saal service karni hai", True, ["DOC-BOND-BR-2627"], None),
        ("Q25", "up me mbbs ke liye kitna bond bharna padta hai", True, ["DOC-BOND-UP-2627"], None),
        ("Q26", "varanasi medical college me admission deposit kitna hai", True, ["DOC-GIMS-VNS-FEE-2627"], "2026-27"),
        ("Q27", "kolkata medical college er mbbs fees koto 2026-27", True, ["DOC-RMCH-KOL-FEE-2627"], "2026-27"),
        ("Q28", "haryana bond er taka koto", True, ["DOC-BOND-HR-2627"], None),
        ("Q29", "kalinga medical college bhubaneswar re seat koto", True, ["DOC-KIMS-BBS-FEE-2627"], "2026-27"),
        ("Q30", "west bengal bond kemon ache", True, ["DOC-BOND-WB-2627"], None),
        ("Q31", "What is the MBBS tuition fee at a medical college in Assam?", False, [], None),
        ("Q32", "Who is the current director of AIIMS New Delhi?", False, [], None),
        ("Q33", "What is the NEET-UG 2027 cut-off for Shanti Medical College, Patna?", False, [], None),
        ("Q34", "What is the annual salary of a junior resident at Kalinga Institute of Medical Sciences?", False, [], None),
        ("Q35", "What is the BDS bond amount in Haryana?", False, [], None),
        ("Q36", "What is the tuition fee for session 2029-30 at Shanti Medical College, Patna?", False, [], "2029-30"),
        ("Q37", "How many MBBS seats does a medical college in Rajasthan have?", False, [], None),
        ("Q38", "What is the placement package for nursing graduates at Rajarhat Medical College?", False, [], None),
        ("Q39", "What is the hostel fee at Saraswati Medical College, Rohtak for 2026-27?", True, ["DOC-SMC-ROH-FEE-2627"], "2026-27"),
        ("Q40", "What is the one-time admission deposit at Deccan Medical College?", True, ["DOC-DMC-HYD-FEE-2627"], "2026-27"),
    ]
    out = []
    for qid, text, answerable, gold, session in q:
        lang = en
        if qid in {"Q21", "Q22", "Q23", "Q24", "Q25", "Q26"}:
            lang = hi
        if qid in {"Q27", "Q28", "Q29", "Q30"}:
            lang = bn
        out.append(
            {
                "qid": qid,
                "question": text,
                "lang": lang,
                "answerable": answerable,
                "gold_docs": gold,
                "session": session,
            }
        )
    return out


SPECIALTIES = ["Critical Care", "General Medicine", "Emergency Medicine", "Radiology", "Community Medicine"]
FIRST = ["A.", "R.", "S.", "M.", "P.", "N.", "K.", "T.", "D.", "V.", "I.", "B.", "G.", "J.", "L."]
SUR = [
    "Bhattacharya", "Iyer", "Kulkarni", "Sen", "Qureshi", "Nair", "Chowdhury",
    "Reddy", "Mukherjee", "Patel", "Rao", "Das", "Verma", "Sinha", "Menon",
    "Ghosh", "Banerjee", "Pillai", "Joshi", "Haldar", "Mishra", "Fernandes",
    "Kaur", "Bose", "Thakur", "Sethi", "Anand", "Dutta", "Sharma", "Agarwal",
]
SKILLS = {
    "Critical Care": ["ventilator management", "ICU protocols", "sepsis", "ECMO", "central lines"],
    "General Medicine": ["internal medicine", "clinical teaching", "diabetes care", "research", "NMC norms"],
    "Emergency Medicine": ["triage", "trauma care", "ACLS", "emergency ultrasound", "poisoning management"],
    "Radiology": ["CT reporting", "MRI basics", "PACS", "ultrasound", "contrast studies"],
    "Community Medicine": ["epidemiology", "biostatistics", "field visits", "teaching", "health survey"],
}
DEGREES = ["MD", "MS", "DNB", "MBBS"]


def build_heldout() -> list[dict]:
    """Held-out questions, written AFTER the retrieval was tuned and never used
    to set a threshold. These are the honest numbers.
    """
    rows = [
        ("H01", "What does Ganges Institute in Varanasi charge for tuition?", True, ["DOC-GIMS-VNS-FEE-2627"], "2026-27", "no 'fee' noun, no programme word"),
        ("H02", "Rohtak MBBS first year total including hostel", True, ["DOC-SMC-ROH-FEE-2627"], "2026-27", "telegraphic, no verb"),
        ("H03", "Haryana bond kitna hai bhai", True, ["DOC-BOND-HR-2627"], None, "slang + code-mix"),
        ("H04", "deccan medical college hyderabad er deposit koto", True, ["DOC-DMC-HYD-FEE-2627"], "2026-27", "roman Bengali word order"),
        ("H05", "Which medical college charges the highest tuition in India?", False, [], None, "comparison across documents"),
        ("H06", "What is the MBBS fee at Kalinga in 2025-26?", True, ["DOC-KIMS-BBS-FEE-2526"], "2025-26", "past session"),
        ("H07", "What is the MBBS bond amount in Odisha?", False, [], None, "no Odisha bond document"),
        ("H08", "Is there a bond at Deccan Medical College?", False, [], None, "no bond document for Telangana"),
        ("H09", "bihar me bond koto", True, ["DOC-BOND-BR-2627"], None, "Bengali question word, Hindi word order"),
        ("H10", "What is the penalty if I leave the course early in Haryana?", True, ["DOC-BOND-HR-2627"], None, "synonyms: penalty, leave"),
        ("H11", "What are the seat upgrade rules in round 3 of counselling?", True, ["DOC-MCC-RULES-2627"], None, "multi-condition question"),
        ("H12", "What is the bus fee for the Jhajjar route?", True, ["DOC-NOISE-TRANSPORT-2627"], "2026-27", "distractor document"),
    ]
    out = []
    for qid, text, answerable, gold, session, note in rows:
        out.append(
            {
                "qid": qid,
                "question": text,
                "lang": "bn-rom" if qid == "H04" else ("hi-rom" if qid in {"H03", "H09"} else "en"),
                "answerable": answerable,
                "gold_docs": gold,
                "session": session,
                "note": note,
            }
        )
    return out


def build_resumes() -> list[dict]:
    import random

    rng = random.Random(20261126)
    resumes = []
    # Deliberate composition: 12 women of 30 (~40%), 6 from tier-1 colleges
    # (~20%), matching the shape of the pilot numbers in Exhibit 4.
    plan = (
        [("Critical Care", 6), ("General Medicine", 7), ("Emergency Medicine", 5),
         ("Radiology", 6), ("Community Medicine", 6)]
    )
    rid = 0
    for specialty, count in plan:
        for _ in range(count):
            rid += 1
            gender = "woman" if rid % 5 in (1, 3) else "man"
            tier = 1 if rid % 5 == 0 else (2 if rid % 2 == 0 else 3)
            degree = "MBBS" if specialty == "Radiology" and rid % 2 == 0 else rng.choice(["MD", "MD", "DNB", "MS"])
            years = rng.choice([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 15])
            gap_type = rng.choice(["none", "none", "none", "maternity", "illness", "caregiving"])
            gap_months = 0 if gap_type == "none" else rng.choice([6, 9, 12, 18])
            skills = rng.sample(SKILLS[specialty], k=rng.choice([2, 3, 3, 4]))
            resumes.append(
                {
                    "rid": f"R{rid:02d}",
                    "name": f"{rng.choice(FIRST)} {SUR[rid - 1]}",
                    "gender": gender,
                    "college_tier": tier,
                    "specialty": specialty,
                    "degree": degree,
                    "relevant_years": years,
                    "years_since_last_role": 0 if gap_type == "none" else round(gap_months / 12, 1),
                    "gap_type": gap_type,
                    "gap_months": gap_months,
                    "skills": skills,
                    "publications": rng.choice([0, 0, 1, 2, 4, 7]),
                    "current_role": rng.choice(
                        ["Senior Resident", "Consultant", "Assistant Professor", "Registrar", "Fellow"]
                    ),
                    "notes": "Registered with the state medical council; teaching experience recorded.",
                }
            )
    return resumes


def build_jobs() -> list[dict]:
    return [
        {
            "jid": "J1", "hospital": "Deccan Institute of Critical Care, Hyderabad",
            "post": "Senior Consultant, Intensive Care", "specialty": "Critical Care",
            "degree_level": "MD", "min_years": 10,
            "required_skills": ["ventilator management", "ICU protocols", "sepsis", "ECMO"],
            "text": "Lead a 24-bed ICU. Must run sepsis and ECMO protocols and supervise residents.",
        },
        {
            "jid": "J2", "hospital": "Shanti Medical College, Patna",
            "post": "Assistant Professor, General Medicine", "specialty": "General Medicine",
            "degree_level": "MD", "min_years": 3,
            "required_skills": ["clinical teaching", "internal medicine", "research"],
            "text": "Teach MBBS students and run the medicine outpatient department under NMC norms.",
        },
        {
            "jid": "J3", "hospital": "Rajarhat Medical College & Hospital, Kolkata",
            "post": "Consultant, Emergency Medicine", "specialty": "Emergency Medicine",
            "degree_level": "MD", "min_years": 5,
            "required_skills": ["triage", "trauma care", "ACLS", "emergency ultrasound"],
            "text": "Run the emergency department across three shifts, including trauma and poisoning cases.",
        },
        {
            "jid": "J4", "hospital": "Kalinga Institute of Medical Sciences, Bhubaneswar",
            "post": "Junior Resident, Radiology", "specialty": "Radiology",
            "degree_level": "MBBS", "min_years": 0,
            "required_skills": ["CT reporting", "PACS"],
            "text": "Support CT and MRI reporting under a consultant radiologist.",
        },
        {
            "jid": "J5", "hospital": "Ganges Institute of Medical Sciences, Varanasi",
            "post": "Assistant Professor, Community Medicine", "specialty": "Community Medicine",
            "degree_level": "MD", "min_years": 2,
            "required_skills": ["epidemiology", "biostatistics", "field visits", "teaching"],
            "text": "Teach community medicine and run rural field visits and health surveys.",
        },
    ]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    docs = build_documents()
    ids = {d["doc_id"] for d in docs}
    assert len(ids) == len(docs), "duplicate doc ids"
    qs = build_questions()
    for q in qs:
        for g in q["gold_docs"]:
            assert g in ids, f"{q['qid']} cites missing {g}"
    (DATA / "documents.json").write_text(json.dumps(docs, indent=1, ensure_ascii=False), encoding="utf-8")
    (DATA / "questions.json").write_text(json.dumps(qs, indent=1, ensure_ascii=False), encoding="utf-8")
    ho = build_heldout()
    for q in ho:
        for g in q["gold_docs"]:
            assert g in ids, f"{q['qid']} cites missing {g}"
    (DATA / "questions_heldout.json").write_text(json.dumps(ho, indent=1, ensure_ascii=False), encoding="utf-8")
    (DATA / "resumes.json").write_text(json.dumps(build_resumes(), indent=1), encoding="utf-8")
    (DATA / "jobs.json").write_text(json.dumps(build_jobs(), indent=1), encoding="utf-8")
    print(f"documents: {len(docs)}  questions: {len(qs)}  resumes: {len(build_resumes())}  jobs: {len(build_jobs())}")
    print(f"unanswerable: {sum(1 for q in qs if not q['answerable'])}, roman-script: {sum(1 for q in qs if q['lang'] != 'en')}")


if __name__ == "__main__":
    main()
