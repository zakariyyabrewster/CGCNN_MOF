import os, re
import logging
import threading
import time
import json
import sys
import signal

class JobMonitorDaemon:
    """Lightweight daemon to monitor fine-tuning jobs in background thread"""
    
    def __init__(self, client, log_dir, poll_interval=60):
        self.client = client
        self.log_dir = log_dir
        self.poll_interval = poll_interval
        self.running = False
        self.thread = None
        
        # Setup logging
        self.log_file = os.path.join(log_dir, 'daemon.log')
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{__name__}.daemon")
    
    def start(self):
        """Start daemon in background thread"""
        if self.running:
            print("Daemon already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info("Job monitor daemon started")
        print(f"Job monitor daemon started (logs: {self.log_file})")
    
    def stop(self):
        """Stop daemon"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.info("Job monitor daemon stopped")
        print("Job monitor daemon stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_jobs()
                time.sleep(self.poll_interval)
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.poll_interval)
    
    def _check_jobs(self):
        """Check all job state files for active jobs"""
        try:
            # Find all job_state.json files recursively
            for root, dirs, files in os.walk(self.log_dir):
                if 'job_state.json' in files:
                    job_file = os.path.join(root, 'job_state.json')
                    self._check_single_job(job_file)
        except Exception as e:
            self.logger.error(f"Error scanning job files: {e}")
    
    def _check_single_job(self, job_file):
        """Check a single job state file"""
        try:
            with open(job_file, 'r') as f:
                state = json.load(f)
            
            job_id = state.get('job_id')
            if not job_id:
                return
            
            # Skip if already completed
            completion_file = os.path.join(os.path.dirname(job_file), f'completion_{job_id}.json')
            if os.path.exists(completion_file):
                return
            
            # Check job status
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            self.logger.info(f"Job {job_id} status: {status}")
            
            # Handle completed jobs
            if status in {"succeeded", "failed", "cancelled"}:
                # Update state with final model
                if status == "succeeded":
                    state['fine_tuned_model'] = job.fine_tuned_model
                
                # Create completion notification
                completion_data = {
                    'job_id': job_id,
                    'status': status,
                    'completed_at': getattr(job, 'finished_at', None),
                    'fine_tuned_model': job.fine_tuned_model if status == "succeeded" else None,
                    'error': getattr(job, 'error', None) if status == "failed" else None
                }
                
                with open(completion_file, 'w') as f:
                    json.dump(completion_data, f, indent=2)
                
                # Update original state file
                with open(job_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                self.logger.info(f"Job {job_id} completed with status: {status}")
                print(f"🎉 Job {job_id} completed! Status: {status}")
                
                if status == "succeeded":
                    print(f"📁 Model: {job.fine_tuned_model}")
                    print(f"📄 Details: {completion_file}")
                    
        except Exception as e:
            self.logger.error(f"Error checking job file {job_file}: {e}")

def setup_monitoring(finetuner, daemon, job_id=None):
    """Setup monitoring choice and handle user selection"""
    
    completion_file = f"{finetuner.config['log_dir']}/completion_{job_id or finetuner.job_id}.json"
    print(f"Background monitoring active")
    print(f"Completion file: {completion_file}")
    print(f"Daemon logs: {daemon.log_file}")
    print("Re-run script later to check results and evaluate")
    
    try:
        while daemon.running:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping daemon...")
    finally:
        daemon.stop()
    sys.exit(0)

def wait_for_completion(finetuner, daemon=None):
    """Wait for job completion and handle cleanup"""
    try:
        status, job_info = finetuner.wait()
        if daemon:
            daemon.stop()
        return status, job_info
    except TimeoutError as e:
        print(f"⏰ Wait timed out: {e}")
        print("💡 You can re-run this script later to resume monitoring")
        if daemon:
            daemon.stop()
        sys.exit(1)

def handle_completed_job(status, finetuner):
    """Handle job completion - success or failure"""
    print(f"\n🏁 Job finished with status: {status}")
    
    if status == "succeeded":
        print(f"🎉 Fine-tuned model ID: {finetuner.fine_tuned_model}")
        print("🔍 Starting evaluation...")
        eval_results = finetuner.eval_jsonl()
        print(f"📊 Evaluation results: {eval_results}")
        return True
    else:
        print(f"❌ Job did not succeed. Status: {status}")
        if status == "failed":
            print("💥 Fine-tuning failed. Check the job details above.")
        elif status == "cancelled":
            print("🚫 Fine-tuning was cancelled.")
        return False

def setup_hpc_monitoring(finetuner, daemon, job_id=None):
    """Setup background monitoring for HPC clusters (always daemon mode)"""
    completion_file = f"{finetuner.config['log_dir']}/completion_{job_id or finetuner.job_id}.json"
    
    print(f"🖥️  HPC Mode: Background monitoring active")
    print(f"📄 Completion file: {completion_file}")
    print(f"📋 Daemon logs: {daemon.log_file}")
    print("🔄 Re-run script later to check results and evaluate")
    print("💡 Job will continue running even if this script exits")
    print("\nPress Ctrl+C to stop daemon and exit...")
    
    try:
        while daemon.running:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping daemon...")
    finally:
        daemon.stop()
    sys.exit(0)

def check_for_completion(finetuner):
    """Check if job is already completed and handle accordingly"""
    completion_file = f"{finetuner.config['log_dir']}/completion_{finetuner.job_id}.json"
    
    if os.path.exists(completion_file):
        with open(completion_file, 'r') as f:
            completion_data = json.load(f)
        
        status = completion_data.get('status')
        print(f"🎯 Job already completed with status: {status}")
        
        if status == "succeeded":
            finetuner.fine_tuned_model = completion_data.get('fine_tuned_model')
            
        return handle_completed_job(status, finetuner)
    
    return None  # Job not completed yet

def setup_signal_handlers(daemon_ref):
    """Setup signal handlers for graceful shutdown with daemon reference"""
    def signal_handler(signum, frame):
        print("\n🛑 Received interrupt signal, shutting down...")
        if daemon_ref and daemon_ref['daemon']:
            daemon_ref['daemon'].stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    return signal_handler

def wait_for_job(finetuner, poll_s=60, timeout_min=180, save_state=True):
    """
    Synchronous waiting for fine-tuning job completion.
    Useful for non-HPC environments where blocking wait is acceptable.
    
    Args:
        finetuner: OpenAI finetuner instance with job_id set
        poll_s: Polling interval in seconds
        timeout_min: Timeout in minutes  
        save_state: Whether to save job state during monitoring
        
    Returns:
        Tuple[str, dict]: (status, job_info)
    """
    import time
    
    # STEP 1: Validate we have a job to monitor
    if not finetuner.job_id:
        raise ValueError("Job ID is not set. Start a job first.")
    
    # STEP 2: Persist job state immediately for crash recovery
    if save_state:
        finetuner._save_job_state()
    
    # STEP 3: Start timing for timeout calculation
    t0 = time.time()
    
    # STEP 4: Main monitoring loop - runs until job completes or times out
    while True:
        try:
            # STEP 5: Query OpenAI API for current job status
            job = finetuner.client.fine_tuning.jobs.retrieve(finetuner.job_id)
            status = job.status  # Can be: queued, running, succeeded, failed, cancelled
            
            # STEP 6: Show progress to user (helpful for long-running jobs)
            print(f"Job status: {status} (elapsed: {(time.time() - t0)/60:.1f} min)")
            
            # STEP 7: Check if job reached a terminal state
            if status in {"succeeded", "failed", "cancelled"}:
                obj = job.to_dict()  # Get full job details for return
                
                # STEP 8: If successful, extract the fine-tuned model ID
                if status == "succeeded":
                    finetuner.fine_tuned_model = obj.get("fine_tuned_model")
                    # STEP 9: Save updated state with model ID for future use
                    if save_state:
                        finetuner._save_job_state()
                else:
                    print(f"Job failed or was cancelled.")
                    print(f"Job details: {obj}")
                    print(f"Status: {status}")
                    print(f"Error message: {obj.get('error', 'No error message available')}")
                    sys.exit(1)

                # STEP 10: Exit loop and return final status
                return status, obj
                
            # STEP 11: Check if we've exceeded timeout limit
            if time.time() - t0 > timeout_min * 60:
                # STEP 12: Inform user they can resume monitoring later
                print(f"Timeout reached. Job ID: {finetuner.job_id} - you can resume monitoring later")
                raise TimeoutError("Job timed out after {} minutes".format(timeout_min))
                
        except Exception as e:
            # STEP 13: Handle network/API errors gracefully - don't crash, just retry
            print(f"Error checking job status: {e}. Retrying in {poll_s} seconds...")
            
        # STEP 14: Wait before next status check to avoid rate limiting
        time.sleep(poll_s)