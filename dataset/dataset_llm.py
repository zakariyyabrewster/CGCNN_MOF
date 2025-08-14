import csv, re, json
import argparse, json, os, re, time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import sys
import os
from pymatgen.core.structure import Structure


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
            "Di": "Di",
            "Dif": "Dif",
            "Df": "Df",
            "CO2_LP": "CO2 uptake at 0.15 bar",
            "CH4_HP": "CH4 uptake at 65 bar",
            "logKH_CO2": "log of Henry's constant for CO2"
        }
        self.prop_to_prompt = self.prop_to_prompt_dict.get(self.prop_name, self.prop_name)

    def _sanitize(self, text: str) -> str:
        if text is None:
            return ""
        t = str(text)
        return t

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
        return {"raw": self._sanitize(s), 
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

    def row_to_ex(self, r: dict, test: bool = False):
        try:
            mofid_str = r["MOFID"]
            y = self._round_label(r[self.prop_name])
            user_payload = self._make_user_payload(mofid_str)
        except Exception:
            if self.drop_bad: return None
            mofid_str, y = r.get("MOFID", ""), float("nan")
            user_payload = json.dumps({"mofid": str(mofid_str), "fields": {}}, separators=(",", ":"))

        # Handle NaN values in completion
        if pd.isna(y):
            target_text = "NaN"
        else:
            target_text = self.num_fmt.format(y)

        system_txt = f"You are a crystallography regression model. Given MOFID metadata, output only {self.prop_to_prompt} in {self.units} as a number (no units, no extra text)."
        
        if not test:
            return {
                "messages": [
                    {"role": "system", "content": system_txt},
                    {"role": "user", "content": user_payload},
                    {"role": "assistant", "content": target_text}
                ]
            }
        else:
            return {
                "messages": [
                    {"role": "system", "content": system_txt},
                    {"role": "user", "content": user_payload},
                ]
            }

    def df_to_jsonl(self, df, jsonl_path, test: bool = False):
        with open(jsonl_path, "w", encoding="utf-8") as out:
            for _, r in df.iterrows():
                example = self.row_to_ex(r, test=test)
                if example is not None:  # Skip None entries when drop_bad=True
                    out.write(json.dumps(example) + "\n")

    def format_for_inference(self, mofid: str) -> str:
        p = self._parse_mofid(mofid)
        return self._make_user_payload(p)

class PromptGenCIF:
    def __init__(self, config):
        self.config = config['prompt-gen']
        self.prop_name = self.config['target_property']
        self.units = self.config['units']
        self.num_precision = self.config['num_precision']
        self.num_fmt = f"{{:.{self.num_precision}f}}"
        self.SEPARATOR = self.config['SEPARATOR']
        self.STOP = self.config['STOP']
        self.drop_bad = self.config['drop_bad']
    
    def _sanitize(self, text: str) -> str:
        if text is None:
            return ""
        t = str(text)
        t = t.replace(self.SEPARATOR, " ")
        t = t.replace(self.STOP, " ")
        return t
    
    def format_cif(self, mofname: str) -> str:
        """
        Format a CIF file name into a prompt string.
        
        Args:
            mofname (str): The name of the MOF file.

        Returns:
            str: A formatted prompt string.
        """
        cif_path = os.path.join(self.config['cif_dir'], f"{mofname}.cif")
        if not os.path.exists(cif_path):
            return {"cell": {"a": None, "b": None, "c": None, "alpha": None, "beta": None, "gamma": None, "volume": None}, "density": None, "formula": {}}
        crystal = Structure.from_file(cif_path)
        lattice = crystal.lattice
        formula = crystal.formula
        cell = {
            "a": round(lattice.abc[0], 3),
            "b": round(lattice.abc[1], 3),
            "c": round(lattice.abc[2], 3),
            "alpha": round(lattice.angles[0], 2),
            "beta": round(lattice.angles[1], 2),
            "gamma": round(lattice.angles[2], 2),
            "volume": round(lattice.volume, 2),
        }
        density = round(crystal.density, 3)
        comp = crystal.composition.get_el_amt_dict()
        formula = {el: int(round(comp[el])) for el in sorted(comp.keys())}
        return {"cell": cell, "density": density, "formula": formula}

    def _make_prompt(self, mofname: str) -> str:

        angstrom = "\u212B"
        degree = "\u00B0"

        formatted_cif = self.format_cif(mofname)
        core = (
            f"Q: Predict {self.prop_name} (units: {self.units}) for MOF.\n"
            f"MOFName: {mofname}\n"
            f"Fields: \n"
            f"    - Lattice: a={formatted_cif['cell']['a']} {angstrom}, b={formatted_cif['cell']['b']} {angstrom}, c={formatted_cif['cell']['c']} {angstrom}, alpha={formatted_cif['cell']['alpha']} {degree}, beta={formatted_cif['cell']['beta']} {degree}, gamma={formatted_cif['cell']['gamma']} {degree}\n"
            f"    - Volume: {formatted_cif['cell']['volume']} {angstrom}^3\n"
            f"    - Density: {formatted_cif['density']} g/cm^3\n"
            f"    - Formula: {', '.join([f'{el}: {count}' for el, count in formatted_cif['formula'].items()])}\n"
        )
        return core + self.SEPARATOR
    
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
            mofname_str = r["MOFName"]
            y = self._round_label(r[self.prop_name])
        except Exception:
            if self.drop_bad: return None
            mofname_str, y = r.get("MOFName", ""), float("nan")

        # Handle NaN values in completion
        if pd.isna(y):
            completion_text = " NaN" + self.STOP
        else:
            completion_text = " " + self.num_fmt.format(y) + self.STOP
            
        return {"prompt": self._make_prompt(mofname_str),
                "completion": completion_text}

    def df_to_jsonl(self, df, jsonl_path):
        with open(jsonl_path, "w", encoding="utf-8") as out:
            for _, r in df.iterrows():
                example = self.row_to_ex(r)
                if example is not None:  # Skip None entries when drop_bad=True
                    out.write(json.dumps(example) + "\n")

    def format_for_inference(self, mofname: str) -> str:
        return self._make_prompt(mofname)


def parse_first_float(text: str, float_re: re.Pattern) -> Optional[float]:
    m = float_re.search(text)
    return float(m.group(0)) if m else None

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

def extract_label(example: dict, stop: Optional[str] = None) -> Optional[float]:
    """
    Extract the label (float) from either:
      - old format: {"completion": " 4.46@@@"}
      - new chat format: {"messages": [{"role": "assistant", "content": "4.46"}]}
    
    Args:
        example (dict): Parsed JSONL row.
        stop (str, optional): Stop token to strip (old format only).
    
    Returns:
        float or None: The extracted label.
    """
    # Case 1: Chat format
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
