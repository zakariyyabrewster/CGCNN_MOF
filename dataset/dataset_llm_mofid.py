import csv, re, json
PROP_NAME = "Di"
UNITS = "angstrom"
NUM_PRECISION = 2
NUM_FMT = f"{{:.{NUM_PRECISION}f}}"

def round_label(x):
    return float(NUM_FMT.format(float(x)))

def parse_mofid(raw: str):
    s = raw.strip().strip('"')
    if "&&" not in s:
        raise ValueError(f"Bad MOFID (no &&): {s}")
    smiles, tail = s.split("&&", 1)              # everything before && is SMILES
    tail = tail.replace(" ", "")
    if "." in tail:
        topo_raw, cat = tail.rsplit(".", 1)
    else:
        topo_raw, cat = tail, "cat0"
    topo = topo_raw.split(",")[0].lower()        # handle "pcu,ERROR"
    if topo in {"", "unknown", "error"}: topo = "unknown"
    cat = cat.lower()
    m = re.search(r"cat\d+", cat)
    cat = m.group(0) if m else ("cat0" if "0" in cat else "cat1" if "1" in cat else "cat0")
    return {"raw": s, "smiles": smiles.strip(), "topology": topo, "catenation": cat}

# mofid = "Cl[Mn][Mn]Cl.[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-]&&tbo.cat0"

# parsed = parse_mofid(mofid)
# print(json.dumps(parsed, indent=2))

def make_prompt(p):
    return (
        f"Q: Predict {PROP_NAME} (units: {UNITS}) for MOF.\n"
        f"MOFID: {p['raw']}\n"
        f"Fields: \n"
        f"    - SMILES: {p['smiles']}\n"
        f"    - Topology: {p['topology']}\n"
        f"    - Catenation: {p['catenation']}\n"
    )

# print(make_prompt(parsed))

def csv_to_jsonl(csv_path, jsonl_path, drop_bad=True):
    with open(csv_path, newline="", encoding="utf-8") as f, \
         open(jsonl_path, "w", encoding="utf-8") as out:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2: continue
            mofid, val = row[0], row[1]
            try:
                p = parse_mofid(mofid)
                y = round_label(val)
            except Exception:
                if drop_bad: continue
                p, y = {"raw": mofid, "smiles": "", "topology": "unknown", "catenation": "cat0"}, float("nan")
            ex = {"prompt": make_prompt(p), "completion": " " + NUM_FMT.format(y)}
            out.write(json.dumps(ex) + "\n")

def row_to_ex(r, drop_bad=True):
    try:
        p = parse_mofid(r["MOFID"])
        y = round_label(r[PROP_NAME])
    except Exception:
        if drop_bad: return None
        p, y = {"raw": r["MOFID"], "smiles": "", "topology": "unknown", "catenation": "cat0"}, float("nan")
    return {"prompt": make_prompt(p),
            "completion": " " + NUM_FMT.format(y)}

def df_to_jsonl(df, jsonl_path):
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for _, r in df.iterrows():
            out.write(json.dumps(row_to_ex(r)) + "\n")


                  

# csv_to_jsonl("test_datasets/mofid_Di.csv", "test_datasets/mofid_Di.jsonl")

data = []
with open("test_datasets/mofid_Di.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        data.append(obj)


