import sys
from dotenv import load_dotenv
from openai import OpenAI
import copy
import os
import yaml
import pandas as pd
import csv
import json
from model.utils import split_data_df
from model.llm_utils import wait_for_job
from dataset.dataset_llm import *
from typing import Optional, Dict, Tuple, List
import time, re
import numpy as np
'''
Finetune OpenAI Model on CoRE2019 Dataset
'''
class OpenAIFinetune:
    def __init__(self, config, api_key):
        '''Initialize OpenAI finetuner'''
        self.config = config

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

        # set base model to gpt-4o-mini
        self.base_model = self.config["finetuner"]["base_model"]
        # set hyperparameters for finetuning
        self.hyperparameters = {
            "n_epochs": self.config["finetuner"]["n_epochs"],
            "batch_size": self.config["finetuner"]["batch_size"],
            "learning_rate_multiplier": self.config["finetuner"]["learning_rate_multiplier"]
        }

        # set suffix for finetuned model
        self.suffix = self.config["finetuner"]["suffix"]

        self.train_file_id: Optional[str] = None
        self.val_file_id: Optional[str] = None
        self.job_id: Optional[str] = None
        self.fine_tuned_model: Optional[str] = None
        self.state_file: Optional[str] = None

        # Create mapping from MOFID to MOFname
        df = pd.read_csv(self.config["lookup_path"])
        self.mofid_to_mofname = dict(zip(df["MOFID"], df["MOFname"]))

        # Create log directory
        os.makedirs(self.config['log_dir'], exist_ok=True)

    def upload_file(self, path: str) -> str:
        '''Upload file to OpenAI API for finetuning'''
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as fp:
            f = self.client.files.create(file=fp, purpose="fine-tune")
        return f.id
    
    def start_job(self):
        '''Start the fine-tuning job'''
        self.train_file_id = self.upload_file(self.config['prompt-gen']['train_jsonl'])

        # Create kwargs for job (model, training_file, validation_file, suffix, method)
        kwargs = dict(
            model=self.base_model,
            training_file=self.train_file_id,
            suffix=self.suffix,
            method={
                "type": "supervised",
                "supervised": {"hyperparameters": self.hyperparameters}
            }
        )
        # Add validation file if available
        val_path = self.config["prompt-gen"].get("val_jsonl")
        if val_path:
            self.val_file_id = self.upload_file(val_path)
            kwargs["validation_file"] = self.val_file_id


        print(kwargs.items())

        # Start the fine-tuning job
        job = self.client.fine_tuning.jobs.create(**kwargs)
        self.job_id = job.id
        print(f"Started job {self.job_id} with training file {self.train_file_id}")
        return job
    
    def _save_job_state(self):
        """Save job state to file for recovery"""
        # PERSISTENCE: Create recoverable state file in log directory
        self.state_file = os.path.join(self.config['log_dir'], 'job_state.json')
        state = {
            'job_id': self.job_id,                    # OpenAI job ID for monitoring
            'fine_tuned_model': self.fine_tuned_model, # Model ID once completed
            'train_file_id': self.train_file_id,      # Uploaded training file ID
            'val_file_id': self.val_file_id          # Uploaded validation file ID
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Job state saved to {self.state_file}")

    def load_job_state(self, job_state_path: Optional[str] = None):
        """Load job state from file for recovery"""
        # Assign state_file if not already set
        if self.state_file is None:
            self.state_file = job_state_path
            if self.state_file is None:
                return False
            
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                
            # Restore IDs
            self.job_id = state.get('job_id')
            self.fine_tuned_model = state.get('fine_tuned_model')
            self.train_file_id = state.get('train_file_id')
            self.val_file_id = state.get('val_file_id')

            print(f"Job state loaded from {self.state_file}")

            return True  # Successfully loaded existing state
        
        return False     # No existing state found

    def generate_prop_val(self, prompt: str, max_completion_tokens: int = 10) -> Optional[float]:
        if not self.fine_tuned_model:
            raise ValueError("Fine-tuned model is not set. Run `start_job` and monitor for completion first.")

        prompt["messages"] = [msg for msg in prompt["messages"] if msg["role"] != "assistant"]
        test_prompt = prompt["messages"]

        print(test_prompt)

        response = self.client.chat.completions.create(
            model=self.fine_tuned_model,
            messages=test_prompt,
            temperature=0,
            max_completion_tokens=max_completion_tokens
        )

        # Use the convenient output_text property
        raw_pred = response.choices[0].message.content.strip()
        try:
            return float(raw_pred)
        except ValueError:
            return None

    def eval_jsonl(self) -> Dict[str, float]:
        """Evaluate the fine-tuned model on the test dataset"""
        test_path = self.config['prompt-gen']['test_jsonl']
        output_path = os.path.join(self.config['log_dir'], 'test_results_{}.csv'.format(self.config['prompt-gen']['target_property']))

        data = read_jsonl(test_path)

        targets, preds, names = [], [], []
        i = 0

        for ex in data:
            label = extract_label(ex)
            if label is None:
                continue

            mofid = extract_mofid(ex)
            mofname = self.mofid_to_mofname.get(mofid, "UNKNOWN")

            pred = self.generate_prop_val(ex)
            if pred is None:
                continue

            targets.append(label)
            preds.append(pred)
            names.append(mofname)

            if i % 10 == 0:
                print(f"{i}/{len(data)} - {mofname}: target={label}, pred={pred}")
            i += 1


        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'target', 'pred'])
            for cif_id, target, pred in zip(names, targets, preds):
                writer.writerow((cif_id, target, pred))

        # Log evaluation results
        print(f"Evaluation results saved to {output_path}")

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your OpenAI API key.")

    property_list = ["Di", "Dif", "Df", "CO2_LP", "CH4_HP", "logKH_CO2"]
    unit_dict = {
            "Di": "angstrom",
            "Dif": "angstrom",
            "Df": "angstrom",
            "CO2_LP": "mmol/g",
            "CH4_HP": "mmol/g",
            "logKH_CO2": "unitless"
        }
    
    master_config = yaml.load(open("config_ft_llm.yaml", "r"), Loader=yaml.FullLoader)

    for target_property in property_list:
        config = copy.deepcopy(master_config)
        config["prompt-gen"]["target_property"] = target_property       
        config["prompt-gen"]["units"] = unit_dict.get(target_property, "unknown")
        
        data_name = config["data_name"]
        seed = config['dataloader']['random_seed']
        config["finetuner"]["suffix"] = config["finetuner"]["suffix"].format(target_property)

        config["log_dir"] = os.path.join(config["log_dir"], "{}_{}_{}_{}".format(config["finetuner"]["base_model"], data_name, seed, target_property))
        # ex. log_dir: training_results/finetuning/LLM_MOF/gpt-4o-mini_MOFID_CoRE2019_1_Di

        config["lookup_path"] = config["lookup_path"].format(target_property)

        if not os.path.exists(config["lookup_path"]):
            raise FileNotFoundError(f"Lookup file not found: {config['lookup_path']}")

        # Load the full dataset
        id_prop_full_df = pd.read_csv(config["lookup_path"])
        # Split into Train, Val, Test DF splits
        train_df, valid_df, test_df = split_data_df(id_prop_full_df, **config["dataloader"])

        config['prompt-gen']['train_jsonl'] = config['prompt-gen']['train_jsonl'].format(target_property)
        config['prompt-gen']['val_jsonl'] = config['prompt-gen']['val_jsonl'].format(target_property)
        config['prompt-gen']['test_jsonl'] = config['prompt-gen']['test_jsonl'].format(target_property)

        prompt_gen = PromptGenMOFID(config)

        # Convert DataFrames to JSONL
        prompt_gen.df_to_jsonl(train_df, config['prompt-gen']['train_jsonl'])
        prompt_gen.df_to_jsonl(valid_df, config['prompt-gen']['val_jsonl'])
        prompt_gen.df_to_jsonl(test_df, config['prompt-gen']['test_jsonl'])

        # Set up Finetuner
        finetuner = OpenAIFinetune(config, api_key)

        print(f"Starting fine-tuning on {target_property}...")

        finetuner.start_job()
        # Wait for job to complete (timeout = 3h, poll = 1m)
        status, job_obj = wait_for_job(finetuner)
        print(f"Job status: {status}")
        print(f"Job details: {job_obj}")
        print(f"Fine-tuned model: {finetuner.fine_tuned_model}")
        print(f"Fine-tuning on {target_property} complete.")
        finetuner.eval_jsonl()
        print(f"Evaluation on {target_property} complete.")

    print("All properties processed.")

if __name__ == "__main__":
    main()
