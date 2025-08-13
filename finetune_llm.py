import sys
from dotenv import load_dotenv
from openai import OpenAI
import os
import yaml
import pandas as pd
import csv
import json
from model.utils import split_data_df, parse_first_float
from model.llm_utils import *
from dataset.dataset_llm import *
from typing import Optional, Dict, Tuple, List
import time, re
'''
Finetune OpenAI Model on CoRE2019 Dataset
'''

class OpenAIFinetuneMOFID:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            api_key=self.config["finetuner"]["OPEN_AI_KEY"]
        )
        self.base_model = self.config["finetuner"]["base_model"]
        self.n_epochs = self.config["finetuner"]["n_epochs"]
        self.suffix = self.config["finetuner"]["suffix"]

        self.train_file_id: Optional[str] = None
        self.val_file_id: Optional[str] = None
        self.job_id: Optional[str] = None
        self.fine_tuned_model: Optional[str] = None

        self.float_re = re.compile(self.config["FLOAT_RE"])
        self.stop = self.config["prompt-gen"]["STOP"]

        df = pd.read_csv(self.config["lookup_path"])
        df["MOFID"] = df["MOFID"].astype(str)
        df["MOFname"] = df["MOFname"].astype(str)

        self.mofid_to_mofname = dict(zip(df["MOFID"], df["MOFname"]))

        os.makedirs(self.config['log_dir'], exist_ok=True)

    def upload_file(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as fp:
            f = self.client.files.create(file=fp, purpose="fine-tune")
        return f.id
    
    def start_job(self):

        self.train_file_id = self.upload_file(self.config['prompt-gen']['train_jsonl'])
        self.val_file_id = self.upload_file(self.config['prompt-gen']['val_jsonl'])

        kwargs = dict(
            model=self.base_model,
            training_file=self.train_file_id,
            n_epochs=self.n_epochs,
            suffix=self.suffix,
            hyperparameters={"n_epochs": self.n_epochs}
        )
        if self.val_file_id:
            kwargs["validation_file"] = self.val_file_id

        job = self.client.fine_tuning.jobs.create(**kwargs)
        self.job_id = job.id
        return self.job_id
    
    def _save_job_state(self):
        """Save job state to file for recovery"""
        # PERSISTENCE: Create recoverable state file in log directory
        state_file = os.path.join(self.config['log_dir'], 'job_state.json')
        state = {
            'job_id': self.job_id,                    # OpenAI job ID for monitoring
            'fine_tuned_model': self.fine_tuned_model, # Model ID once completed
            'train_file_id': self.train_file_id,      # Uploaded training file ID
            'val_file_id': self.val_file_id          # Uploaded validation file ID
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Job state saved to {state_file}")
    
    def load_job_state(self):
        """Load job state from file for recovery"""
        # RECOVERY: Restore job state from previous session
        state_file = os.path.join(self.config['log_dir'], 'job_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            # Restore all job-related IDs from saved state
            self.job_id = state.get('job_id')
            self.fine_tuned_model = state.get('fine_tuned_model')
            self.train_file_id = state.get('train_file_id')
            self.val_file_id = state.get('val_file_id')
            print(f"Job state loaded from {state_file}")
            return True  # Successfully loaded existing state
        return False     # No existing state found
    
    def check_job_status(self):
        """Check current job status without waiting"""
        # QUICK CHECK: Get status once without polling loop
        if not self.job_id:
            raise ValueError("Job ID is not set.")
        job = self.client.fine_tuning.jobs.retrieve(self.job_id)
        return job.status, job.to_dict()

    def generate_prop_val(self, prompt: str, max_output_tokens: int = 12) -> Optional[float]:
        if not self.fine_tuned_model:
            raise ValueError("Fine-tuned model is not set. Run `start_job` and monitor for completion first.")
        
        response = self.client.responses.create(
            model=self.fine_tuned_model,
            input=prompt,
            temperature=0,
            max_output_tokens=max_output_tokens,
            stop=[self.stop]
        )

        # Use the convenient output_text property
        text_content = response.output_text
        if text_content:
            # parse_first_float uses regex to extract just the numeric part,
            # automatically ignoring the stop sequence
            return parse_first_float(text_content.strip(), self.float_re)
        
        return None
    
    def eval_jsonl(self) -> Dict[str, float]:
        data = read_jsonl(self.config['prompt-gen']['test_jsonl'])
        targets = []
        preds = []
        names = []
        for ex in data:
            label = extract_label(ex["completion"], stop=self.stop)
            if label is None:
                continue
            mofid = extract_mofid(ex["prompt"])
            mofname = self.mofid_to_mofname.get(mofid, "UNKNOWN")

            pred = self.generate_prop_val(ex["prompt"])
            if pred is None:
                continue

            targets.append(label)
            preds.append(pred)
            names.append(mofname)

        with open(os.path.join(self.config['log_dir'], 'test_results_{}.csv'.format(self.config['prompt-gen']['target_property'])), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'target', 'pred'])
            for cif_id, target, pred in zip(names, targets, preds):
                writer.writerow((cif_id, target, pred))

        # Calculate and return basic metrics
        if targets and preds:
            import numpy as np
            mae = np.mean(np.abs(np.array(targets) - np.array(preds)))
            rmse = np.sqrt(np.mean((np.array(targets) - np.array(preds))**2))
            return {"mae": mae, "rmse": rmse, "n_samples": len(targets)}
        else:
            return {"mae": float('nan'), "rmse": float('nan'), "n_samples": 0}

# if __name__ == "__main__":
#     config = yaml.load(open("config_ft_llm.yaml", "r"), Loader=yaml.FullLoader)
#     target_property = config["prompt-gen"]["target_property"]
#     data_name = config["data_name"]
#     seed = config['dataloader']['random_seed']
#     config["finetuner"]["suffix"] = config["finetuner"]["suffix"].format(target_property)
#     config["log_dir"] = os.path.join(config["log_dir"].format(config["mof_representation"]), "{}_{}_{}_{}_{}".format(config["finetuner"]["base_model"], config["mof_representation"], data_name, seed, target_property))
#     # ex. log_dir: training_results/finetuning/LLM_MOFID/gpt-4o-mini_MOFID_CoRE2019_1_Di

#     config["lookup_path"] = config["lookup_path"].format(target_property)

#     if not os.path.exists(config["lookup_path"]):
#         raise FileNotFoundError(f"Lookup file not found: {config['lookup_path']}")

#     # Load the full dataset
#     id_prop_full_df = pd.read_csv(config["lookup_path"])
#     # Split into Train, Val, Test DF splits
#     train_df, valid_df, test_df = split_data_df(id_prop_full_df, **config["dataloader"])
#     print(f"Train size: {len(train_df)}, Validation size: {len(valid_df)}, Test size: {len(test_df)}")

#     config['prompt-gen']['train_jsonl'] = config['prompt-gen']['train_jsonl'].format(target_property)
#     config['prompt-gen']['val_jsonl'] = config['prompt-gen']['val_jsonl'].format(target_property)
#     config['prompt-gen']['test_jsonl'] = config['prompt-gen']['test_jsonl'].format(target_property)

#     if config["mof_representation"] == "MOFID":
#         prompt_gen = PromptGenMOFID(config)
#     if config["mof_representation"] == "CIF":
#         prompt_gen = PromptGenCIF(config)

#     # Convert DataFrames to JSONL
#     prompt_gen.df_to_jsonl(train_df, config['prompt-gen']['train_jsonl'])
#     prompt_gen.df_to_jsonl(valid_df, config['prompt-gen']['val_jsonl'])
#     prompt_gen.df_to_jsonl(test_df, config['prompt-gen']['test_jsonl'])

#     # Set up Finetuner
#     load_dotenv() # load API keys
#     api_key = os.getenv("OPEN_AI_KEY")
#     if not api_key:
#         raise ValueError("OPEN_AI_KEY not found in environment variables. Please create a .env file with your OpenAI API key.")
#     config["finetuner"]["OPEN_AI_KEY"] = api_key

#     if config['mof_representation'] == "MOFID":
#         finetuner = OpenAIFinetuneMOFID(config)
    # elif config['mof_representation'] == "cif":
    #     finetuner = OpenAIFinetuneCIF(config)

    # # Setup signal handlers with daemon reference
    # daemon_ref = {'daemon': None}
    # setup_signal_handlers(daemon_ref)

    # # Main execution logic - supports both HPC and local execution
    # try:
    #     # Check if we have an existing job
    #     if finetuner.load_job_state():
    #         print("🔄 Resuming existing fine-tuning job...")
    #         status, job_info = finetuner.check_job_status()
    #         print(f"📋 Current job status: {status}")
            
    #         # Check if job is already completed
    #         if check_for_completion(status, finetuner):
    #             sys.exit(0)  # Successfully handled completed job
    #         else:
    #             # Job still running - setup monitoring
    #             print("⏳ Job still running...")
    #             daemon_ref['daemon'] = JobMonitorDaemon(finetuner.client, config['log_dir'])
    #             daemon_ref['daemon'].start()
                
    #             if setup_monitoring(finetuner, daemon_ref['daemon']):
    #                 # User chose synchronous waiting
    #                 status, job_info = wait_for_completion(finetuner, daemon_ref['daemon'])
                
    #     else:
    #         # Start new job
    #         print("🚀 Starting new fine-tuning job...")
    #         job_id = finetuner.start_job()
    #         print(f"✅ Fine-tuning job started with ID: {job_id}")
            
    #         # Setup monitoring
    #         daemon_ref['daemon'] = JobMonitorDaemon(finetuner.client, config['log_dir'])
    #         daemon_ref['daemon'].start()
            
    #         if setup_monitoring(finetuner, daemon_ref['daemon'], job_id):
    #             # User chose synchronous waiting
    #             status, job_info = wait_for_completion(finetuner, daemon_ref['daemon'])

    #     # Handle completion (only reached if synchronous waiting was chosen)
    #     success = handle_completed_job(status, finetuner)
    #     sys.exit(0 if success else 1)
        
    # except Exception as e:
    #     print(f"💥 Unexpected error: {e}")
    #     if daemon_ref['daemon']:
    #         daemon_ref['daemon'].stop()
    #     sys.exit(1)
