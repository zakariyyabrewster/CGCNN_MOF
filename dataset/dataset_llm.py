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

    def _make_prompt(self, mofid_str: str) -> str:
        p = self._parse_mofid(mofid_str)
        core = (
            f"Q: Predict {self.prop_name} (units: {self.units}) for MOF.\n"
            f"MOFID: {p['raw']}\n"
            f"Fields: \n"
            f"    - SMILES: {p['smiles']}\n"
            f"    - Topology: {p['topology']}\n"
            f"    - Catenation: {p['catenation']}\n"
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
            mofid_str = r["MOFID"]
            y = self._round_label(r[self.prop_name])
        except Exception:
            if self.drop_bad: return None
            mofid_str, y = r.get("MOFID", ""), float("nan")

        # Handle NaN values in completion
        if pd.isna(y):
            completion_text = " NaN" + self.STOP
        else:
            completion_text = " " + self.num_fmt.format(y) + self.STOP
            
        return {"prompt": self._make_prompt(mofid_str),
                "completion": completion_text}

    def df_to_jsonl(self, df, jsonl_path):
        with open(jsonl_path, "w", encoding="utf-8") as out:
            for _, r in df.iterrows():
                example = self.row_to_ex(r)
                if example is not None:  # Skip None entries when drop_bad=True
                    out.write(json.dumps(example) + "\n")

    def format_for_inference(self, mofid: str) -> str:
        p = self._parse_mofid(mofid)
        return self._make_prompt(p)

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

def extract_label(completion_text: str, stop: str) -> Optional[float]:
    t = completion_text
    if t.endswith(stop):
        t =  t[: -len(stop)]
    t = t.strip()
    try:
        return float(t)
    except ValueError:
        return None

def extract_mofid(prompt: str) -> str:
    _MOFID_LINE_RE = re.compile(r"(?mi)^\s*MOFID\s*:\s*(.+?)\s*$")
    """
    Extract the MOFID from a JSONL example.
    
    Args:
        ex (Dict[str, str]): A dictionary representing a JSONL example.

    Returns:
        str: The MOFID string.
    """
    m = _MOFID_LINE_RE.search(prompt)
    if not m:
        raise ValueError(f"MOFID not found in prompt: {prompt}")
    mofid = m.group(1).strip().strip('"')
    return mofid