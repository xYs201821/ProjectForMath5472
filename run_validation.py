"""
This script is used for sampling images from the model using the prompts provided in the JSON file of the validation dataset.
"""
import json
import subprocess
import sys
from time import sleep
import glob
def run_program_with_json(json_file_path, program_to_run):
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    active_processes = []
    for itr, entry in enumerate(data):
        if glob.glob(f"validation/output/vsd/*{itr}.png"):
            print("Skipping ", entry['caption'], " File already exists!")
            continue
        # Check if output file already exists
        # Wait if we already have max_concurrent processes running
        while len(active_processes) >= 3:
            # Update active_processes list by checking each process
            active_processes = [p for p in active_processes if p.poll() is None]
            if len(active_processes) >= 3:
                sleep(1)  # Only sleep if we still need to wait
        
        # Extract relevant information
        prompt = entry['caption']
        objects = ','.join(entry['objects'])
        print(prompt, objects)
        
        try:
            # Start new process
            process = subprocess.Popen([
                sys.executable,
                program_to_run,
                '--prompt', prompt,
                '--iter', f"{itr}",
                '--config', 'validation_config.yaml'
            ])
            active_processes.append(process)
            
        except Exception as e:
            print(f"Error processing {prompt}: {e}")
            continue
    
    # Wait for remaining processes to complete
    for process in active_processes:
        process.wait()

json_file_path = "validation/prompts.json"
program_to_run = "main.py"  # Replace with your program's name
run_program_with_json(json_file_path, program_to_run)