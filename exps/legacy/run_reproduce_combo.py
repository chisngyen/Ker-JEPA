import subprocess
import os
import sys
import time
import json

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Run using the same python interpreter
        process = subprocess.Popen([sys.executable, script_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT, 
                                   text=True,
                                   bufsize=1,
                                   universal_newlines=True)
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n✅ SUCCESS: {script_path} (Took {elapsed/60:.2f} mins)")
            return True
        else:
            print(f"\n❌ FAILED: {script_path} (Return code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"\n⚠️ ERROR unexpected: {str(e)}")
        return False

def main():
    print("🔥 KER-JEPA MASTER REPRODUCTION COMBO 🔥")
    print("Optimization: H100 (Max-Autotune) | Precision: BF16")
    
    scripts = [
        "exp01_ksd_full_table.py", # Target: 91.90%
        "exp02_mmd_variants.py",   # Target: 91.29%
        "exp03_lejepa_official.py" # Target: 91.13%
    ]
    
    results = {}
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    
    for script in scripts:
        full_path = os.path.join(repo_dir, script)
        if os.path.exists(full_path):
            success = run_script(full_path)
            results[script] = "SUCCESS" if success else "FAILED"
        else:
            print(f"⏩ SKIPPING: {script} (File not found)")
            results[script] = "NOT_FOUND"

    print(f"\n{'='*60}")
    print("📊 FINAL REPRODUCTION SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(results, indent=4))
    
    with open("reproduction_summary.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSummary saved to: reproduction_summary.json")

if __name__ == "__main__":
    main()
