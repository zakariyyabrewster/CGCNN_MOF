import os, re, json
from typing import Dict, List, Optional, Tuple
import pandas as pd


'''
Generate JSONL datasets to input as training/validation/test datasets for OpenAI Regression Task
'''

class PromptGenMOFID:
    def __init__(self, config):

        self.config = config['prompt-gen']
        self.prop_name = self.config['target_property']
        self.units = self.config['units']
        self.num_precision = self.config['num_precision']
        self.num_fmt = f"{{:.{self.num_precision}f}}"
        self.drop_bad = self.config['drop_bad']
        self.prop_to_prompt_dict = {
            "Di": "Di", "Dif": "Dif", "Df": "Df",
            "CO2_LP": "CO2 uptake at 0.15 bar",
            "CH4_HP": "CH4 uptake at 65 bar",
            "logKH_CO2": "log Henry's constant for CO2"
        }
        self.prop_to_prompt = self.prop_to_prompt_dict.get(self.prop_name, self.prop_name)

    def _sanitize(self, text: str) -> str:
        return "" if text is None else str(text)

    def _parse_mofid(self, raw: str):
        s = raw.strip().strip('"')
        if "&&" not in s:
            raise ValueError(f"Bad MOFID (no &&): {s}")
        smiles, tail = s.split("&&", 1)              # everything before && is SMILES
        tail = tail.replace(" ", "")
        if "." in tail:
            topo_raw, cat = tail.rsplit(".", 1)
        else:
            topo_raw, cat = tail, "cat0"
        tokens = [t.strip().lower() for t in topo_raw.split(",") if t.strip()]
        bad = {"", "unknown", "error", "na", "n/a", "none", "null"}
        topo = next((t for t in tokens if t not in bad), "unknown")
        cat = cat.lower()
        m = re.search(r"cat\d+", cat)
        cat = m.group(0) if m else ("cat0" if "0" in cat else "cat1" if "1" in cat else "cat0")
        return {
            "raw": self._sanitize(s), 
            "smiles": self._sanitize(smiles.strip()), 
            "topology": self._sanitize(topo), 
            "catenation": self._sanitize(cat)
            }

    def _make_user_payload(self, mofid_str: str) -> str:
        p = self._parse_mofid(mofid_str)
        obj = {
            "mofid": p["raw"],
            "fields": {
                "smiles": p["smiles"],
                "topology": p["topology"],
                "catenation": p["catenation"],
            }
        }
        # compact, stable order
        return json.dumps(obj, separators=(",", ":"), sort_keys=False)
    
    def _round_label(self, label: float) -> float:
        if label is None or pd.isna(label):
            return float("nan")
        try:
            return round(float(label), self.num_precision)
        except Exception as e:
            print(f"Error rounding label {label}: {e}")
            return float("nan")

    def row_to_ex(self, r: dict):
        try:
            mofid_str = r["MOFID"]
            y = self._round_label(r[self.prop_name])
            user_payload = self._make_user_payload(mofid_str)
        except Exception:
            if self.drop_bad: 
                return None
            mofid_str = r.get("MOFID", "")
            y = float("nan")
            user_payload = json.dumps({"mofid": str(mofid_str), "fields": {}}, separators=(",", ":"))

        # Handle NaN values in completion
        if pd.isna(y):
            return None if self.drop_bad else {
            "messages": [
                {"role": "system", "content": f"You are a crystallography regression model. Given MOFID metadata, output only {self.prop_to_prompt} in {self.units} as a number (no units, no extra text)."},
                {"role": "user", "content": user_payload},
                {"role": "assistant", "content": "NaN"}  # keep only if you really want to audit later
            ]
        }

        target_txt = self.num_fmt.format(y)

        system_txt = f"You are a crystallography regression model. Given MOFID metadata, output only {self.prop_to_prompt} in {self.units} as a number (no units, no extra text)."
        
        return {
            "messages": [
                {"role": "system", "content": system_txt},
                {"role": "user", "content": user_payload},
                {"role": "assistant", "content": target_txt}
            ]
        }

    def df_to_jsonl(self, df, jsonl_path):
        with open(jsonl_path, "w", encoding="utf-8", newline="\n") as out:
            for _, r in df.iterrows():
                example = self.row_to_ex(r)
                if example is not None:  # Skip None entries when drop_bad=True
                    out.write(json.dumps(example, separators=(",", ":")) + "\n")

def read_jsonl(path: str) -> List[Dict[str, str]]:
    """
    Read a JSONL file and return a list of dictionaries.
    
    Args:
        path (str): The path to the JSONL file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing the JSON objects.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def extract_label(example: dict) -> Optional[float]:
    """
    Extract the label (float) from {"messages": [{"role": "assistant", "content": "4.46"}]}
    
    Args:
        example (dict): Parsed JSONL row.

    Returns:
        float or None: The extracted label.
    """
    
    if "messages" in example:
        assistant_msg = next((m for m in example["messages"] if m.get("role") == "assistant"), None)
        if not assistant_msg:
            return None
        t = assistant_msg["content"].strip()
        try:
            return float(t)
        except ValueError:
            return None

def extract_mofid(example: dict) -> str:
    """
    Extract MOFID from the new JSONL chat training format.

    Args:
        example (dict): Parsed JSONL row.

    Returns:
        str: MOFID string.
    """
    # Find user message
    user_msg = next((m for m in example.get("messages", []) if m.get("role") == "user"), None)
    if not user_msg:
        raise ValueError(f"No user message found in example: {example}")

    # Parse user content (stored as JSON string)
    try:
        user_content = json.loads(user_msg["content"])
    except json.JSONDecodeError as e:
        raise ValueError(f"User content is not valid JSON: {user_msg['content']}") from e

    mofid = user_content.get("mofid")
    if not mofid:
        raise ValueError(f"MOFID not found in user content: {user_content}")

    return mofid.strip()
