import subprocess
from pathlib import Path


if __name__ == '__main__':
    root_dir = Path(__file__).parents[2]

    # Path to your SLURM script
    slurm_script = f"{root_dir}/bash_scripts/hyperparameter_search.sh"
    exclude_nodes = "gpu-380-[10,12,14]"  # Example node exclusion pattern, here I'm excluding nodes that are not compatible with CUDA 12.9

    try:
        result = subprocess.run(["sbatch", f"--exclude={exclude_nodes}", slurm_script], capture_output=True, text=True, check=True)
        print("Job submitted successfully.")
        print("SLURM response:", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Failed to submit SLURM job.")
        print("Error output:", e.stderr)