# Cluster Cheat Sheet — DMML Course

A single-page reference for the course SLURM cluster. Keep this handy.

---

## First-time setup

### 1. Generate an SSH key (one-time, on your laptop)

```bash
ssh-keygen -t ed25519 -C "your.email@university.edu"
# Press Enter to accept default location (~/.ssh/id_ed25519)
# Choose a passphrase (recommended) or leave empty
```

### 2. Send your **public** key to the cluster admin

```bash
cat ~/.ssh/id_ed25519.pub
# Copy the output and email it to: [admin contact]
# NEVER send the private key (id_ed25519 without .pub)
```

### 3. Add a convenience entry to `~/.ssh/config`

```
Host dmml
    HostName cluster.address.example.edu
    User your_username
    IdentityFile ~/.ssh/id_ed25519
```

Now `ssh dmml` is enough to log in.

### 4. Verify

```bash
ssh dmml
hostname    # should print the login node name
exit
```

---

## Run something now (interactive) — your day-one escape hatch

No job scripts needed. If your laptop is too weak to run an exercise, do it on a
compute node interactively:

```bash
ssh dmml                              # log in to the cluster
srun --pty bash                       # grab an interactive shell on a compute node
source ~/dmml_env/bin/activate        # activate the shared course environment
python my_script.py                   # ...and run your code, right there on the cluster
```

That is the *only* cluster workflow you need for Weeks 1–3. Batch jobs (`sbatch`,
job scripts, queues) are introduced in Week 4 once grid search makes them worth it.

---

## The five SLURM commands you actually need

| Command | What it does |
|---|---|
| `sbatch job.slurm` | Submit a batch job. Returns a job ID. |
| `squeue -u $USER` | Show your queued and running jobs. |
| `scancel <job_id>` | Cancel a job (queued or running). |
| `sacct -j <job_id>` | Job history, including completed jobs. |
| `srun --pty bash` | Open an interactive shell on a compute node. |

For interactive GPU debugging:
```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 --pty bash
```

---

## Minimal CPU job script

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=logs/%x_%j.out      # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00              # HH:MM:SS — be realistic
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Environment setup
source ~/dmml_env/bin/activate       # or: module load python/3.11

# Run
python my_script.py
```

Submit with: `sbatch job.slurm`

---

## Minimal GPU job script

```bash
#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1                 # request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source ~/dmml_env/bin/activate

# Verify GPU is visible — fail loudly if not
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'"

python train_cnn.py
```

---

## File transfer

```bash
# Laptop → cluster
scp my_data.csv dmml:~/project/

# Cluster → laptop
scp dmml:~/project/results.json .

# Whole directory
scp -r dmml:~/project/results/ ./local_results/
```

For larger or syncing workflows, use `rsync`:
```bash
rsync -avz ~/project/ dmml:~/project/
```

---

## The good-citizen rules

1. **Request what you need, not the max.** Asking for 64 CPUs and 256 GB when you need 4 and 8 makes you wait longer *and* blocks others.
2. **Set realistic time limits.** If your job runs 30 minutes, ask for 1 hour, not 24. SLURM schedules short jobs faster.
3. **Cancel failed jobs.** Don't leave broken jobs sitting in the queue.
4. **Don't hoard GPUs.** Especially for interactive sessions — close them when you're not actively using them.
5. **Don't poll `squeue` in a loop.** A watch every few seconds is enough; tighter loops hammer the scheduler.
6. **Test small before submitting big.** Run on a subset locally or in an interactive session before launching a 12-hour sweep.

---

## Top three troubleshooting cases

### Job stuck in `PD` (pending) state forever

```bash
squeue -j <job_id> -o "%i %T %r"   # shows reason
```

Usually one of:
- Requesting more resources than any node has (`PartitionConfig`)
- Asking for a GPU when none are free (`Resources`)
- Hit your user/group fair-share limit (`AssocGrp...`)

**Fix:** lower resource request or wait. Don't resubmit repeatedly — that won't help.

### Job fails immediately, empty or weird output

Check the `.err` file first:
```bash
cat logs/my_job_12345.err
```

Most common causes:
- Virtual environment not activated → `ModuleNotFoundError`
- Wrong working directory → `FileNotFoundError`
- `#SBATCH` directives parsed as bash comments because of a typo (e.g., `# SBATCH` with a space)

### Training runs but is incredibly slow on a "GPU" job

You forgot to verify the GPU. PyTorch falls back to CPU silently. Always include:

```python
import torch
assert torch.cuda.is_available(), "CUDA not available — check --gres in SLURM script"
device = torch.device("cuda")
model = model.to(device)
```

---

## Course conventions

- **Project working directory:** `~/dmml_project/`
- **Shared env:** `~/dmml_env/` (instructions on the course page)
- **Logs:** keep them in `logs/` subdirectory — `.gitignore` it
- **Job time limit policy:** course jobs ≤ 4h CPU, ≤ 6h GPU (request extension if needed)
- **Help:** post in the course channel (`#cluster`) before emailing — others have hit the same issue

---

## One-line emergency commands

```bash
scancel -u $USER              # cancel all your jobs (use with care)
squeue -u $USER -h | wc -l    # how many jobs do I have running/queued
sinfo -o "%P %a %l %D %t"     # what nodes/partitions exist and their state
sacct -j <jobid> -o JobID,State,Elapsed,MaxRSS,ReqMem   # post-mortem on a job
```
