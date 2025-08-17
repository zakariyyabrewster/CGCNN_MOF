import time, sys


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
    
    if not finetuner.job_id:
        raise ValueError("Job ID is not set. Start a job first.")
    
    if save_state:
        finetuner._save_job_state()
    
    # Start timing for timeout
    t0 = time.time()
    
    while True:
        try:
            job = finetuner.client.fine_tuning.jobs.retrieve(finetuner.job_id)
            status = getattr(job, "status", None) or job.get("status") 
            # status in {queued, running, succeeded, failed, cancelled}
            
            print(f"Job status: {status} (elapsed: {(time.time() - t0)/60:.1f} min)")
            
            # STEP 7: Check if job reached a terminal state
            if status in {"succeeded", "failed", "cancelled"}:
                obj = job.to_dict() if hasattr(job, "to_dict") else dict(job)
                
                # extract the fine-tuned model ID
                if status == "succeeded":
                    finetuner.fine_tuned_model = obj.get("fine_tuned_model")
                    # STEP 9: Save updated state with model ID for future use
                    if save_state:
                        finetuner._save_job_state()
                else:
                    print(f"\nJob did not succeed. Final status: {status.upper()}")
                    if "error" in obj and obj["error"]:
                        print(f"Error message: {obj['error']}")
                    else:
                        print("No explicit error message provided by API.")

                # STEP 10: Exit loop and return final status
                return status, obj
                
            # STEP 11: Check if we've exceeded timeout limit
            if time.time() - t0 > timeout_min * 60:
                # STEP 12: Inform user they can resume monitoring later
                print(f"Timeout reached. Job ID: {finetuner.job_id} - you can resume monitoring later")
                raise TimeoutError("Job timed out after {} minutes".format(timeout_min))
                
        except TimeoutError as e:
            # let caller decide what to do on timeout
            raise TimeoutError(
                "Job timed out after {} minutes".format(timeout_min)
            ) from e
        except KeyboardInterrupt:
            print("\nInterrupted. You can resume monitoring later using the saved job_id.")
            raise
        except Exception as e:
            # transient error; retry after a pause
            print(f"Error checking job status: {e}. Retrying in {poll_s} seconds...")
            
        # STEP 14: Wait before next status check to avoid rate limiting
        time.sleep(poll_s)