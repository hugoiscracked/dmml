# Cluster Integration — Lecture Track Insertions

These are the three cluster touchpoints to splice into the main lecture track. Each is written as a drop-in addition (or replacement) for the relevant section, with the surrounding context noted so you know where it fits.

---

## Touch 1 — Week 1, Lecture 1 (Environment Setup)

**Where it goes:** Append to the end of W1 Lecture 1, after beat 12 ("A word on Git") and before the lecture-end transition. Add as new beats 13–16. Adjust the W1 Computer Exercise to include the cluster login task.

### New beats

13. **Introducing the course cluster.** This course provides access to a SLURM-managed compute cluster of decommissioned machines. *Note:* state the rationale explicitly — (a) no hardware discrimination, everyone has the same compute available; (b) SLURM and cluster workflows are an industry-standard skill the university doesn't otherwise teach; (c) the project in W11–W12 will require non-trivial compute that local laptops can't provide.
14. **What you'll do today.** Three steps, all part of the W1 hand-in: generate an SSH key, log in to the cluster, run `hostname` and paste the output. *Note:* this is pure plumbing — no SLURM batch jobs yet. The goal is to surface account/network problems eleven weeks before they would otherwise bite.
15. **An escape hatch from day one.** Demo a single interactive session: `srun --pty bash` → activate the shared env → run a simple script and watch it execute on a compute node. *Note:* this is deliberately the *only* execution mode we teach today. It lets a student with a weak laptop run every subsequent exercise on the cluster interactively, while the full batch/`sbatch` workflow waits until W4 where grid search makes it self-motivating. Keep it to one "it ran on a remote machine" moment — don't introduce `#SBATCH`, queues, or job scripts yet.
16. **Where to get help.** Cheat sheet handout (link/location), office hours this week specifically reserved for cluster access issues, TA contact. *Pitfall:* don't try this at 11pm the night before the W2 hand-in — start in the first 48 hours so issues surface while support is available.

### Modification to W1 Computer Exercise

Add a third bullet:

- **Cluster access:** Generate an SSH key pair, register your public key with the cluster admin (per handout), log in, and run `hostname > w1_cluster_check.txt`. Submit the file with the rest of your W1 exercise.

### Why this placement

The environment-setup mindset is already active — students are installing Python, configuring VS Code, and dealing with virtual environments. Adding "log into a remote machine" fits the theme. No SLURM concepts are introduced; this is purely about getting credentials working.

---

## Touch 2 — Week 4, Computer Exercise (SVM Grid Search)

**Where it goes:** This becomes a substantial expansion of the W4 computer exercise. The first of the two exercise blocks is now a SLURM walkthrough; the second is the SVM grid search run on the cluster.

### Suggested structure for the W4 exercise sessions

#### Block 1 — SLURM walkthrough (≈2h)

The pedagogical hook: grid search is embarrassingly parallel and slow. This is the moment to introduce the tool that fixes both problems.

1. **The motivation.** Show a grid search timing estimate on a laptop. SVM with C ∈ {0.01, 0.1, 1, 10, 100} × γ ∈ {0.001, 0.01, 0.1, 1} × 5-fold CV = 100 fits. On the digits dataset this is tolerable; on anything bigger it isn't. *Note:* frame the cluster as "the natural home for this kind of work," not as an arbitrary tool.
2. **What SLURM is.** A workload manager. You describe a job (resources needed, command to run), submit it to a queue, SLURM finds a machine and runs it. *Note:* the queue is shared — being a good citizen matters.
3. **The anatomy of a job script.** Walk through `submit_cpu.slurm` line by line: shebang, `#SBATCH` directives (job name, output, time, cpus, memory), environment setup, the actual command. *Note:* the `#SBATCH` lines look like comments to bash — they're directives to SLURM. Confusion magnet.
4. **The five commands you actually need.** `sbatch`, `squeue`, `scancel`, `sacct`, `srun --pty bash` for interactive sessions. Demo each.
5. **Live demo — a trivial job.** Submit a job that runs `hostname && sleep 30 && echo done`. Watch it appear in `squeue`. Check the output file. *Note:* demystify the loop before adding ML complexity.
6. **From script to grid search.** Show the same SVM grid search structured as a SLURM job. Same Python code, wrapped in a job script. Discuss what `--cpus-per-task` should be (matches `n_jobs` in `GridSearchCV`).
7. **Output handling.** Where stdout/stderr go. How to retrieve results (`scp`, shared filesystem, or commit to git from the cluster). *Pitfall:* writing to `$HOME` works; writing to `/tmp` doesn't persist; writing huge files anywhere makes you unpopular.
8. **The good-citizen rules.** Request what you need, not the max. Set realistic time limits. Cancel failed jobs. Don't poll `squeue` in a tight loop. *Note:* with ~25 project groups eventually hitting this cluster, etiquette is not optional.
9. **Troubleshooting the top three failures.** (a) Job stuck in `PD` (pending) state — usually requesting more resources than available; (b) job fails immediately — usually environment not loaded or path wrong; (c) job runs forever — missing time limit or infinite loop. Show how to diagnose each.
10. **Interactive vs batch.** `srun --pty bash` for debugging, `sbatch` for production. Most students will want interactive for development, batch for the actual grid search.

#### Block 2 — SVM grid search on the cluster (≈2h)

11. **The task.** Take the SVM exercise from the W4 lecture — digits dataset, linear and RBF kernels, grid search over C and γ. Run it on the cluster.
12. **Required deliverable.** The grid search must run as a SLURM batch job. Hand in: the job script, the SLURM output file, the best hyperparameters found, and CV scores.
13. **Bonus exploration.** Compare wall-clock time of the same grid search local vs. cluster. Discuss what they observe. *Note:* on small datasets local can win — the cluster's value is on bigger problems and parallel sweeps, not on every job.

### Why this placement

Week 4 is when students first feel the pain of hyperparameter tuning. They've just learned grid search and CV. The cluster solves a problem they have *right now*, not one they imagine they'll have later. Forcing cluster submission for the hand-in guarantees every student has run at least one real job before week 11.

It also leaves seven weeks of optional cluster use (W5–W10 exercises) for them to internalize the workflow on lower-stakes work, before the project where it actually matters.

---

## Touch 3 — Week 9, Computer Exercise (CNNs and GPUs)

**Where it goes:** Add a focused intermezzo at the start of the W9 exercise, before students dive into the Fashion-MNIST CNN task.

### Suggested structure (~20–30 min at the top of the W9 exercise)

1. **The new resource: GPUs.** Until now, everything ran on CPU. CNN training on CPU is painful — Fashion-MNIST on a laptop CPU is doable, but anything realistic isn't. *Note:* this is the second pain point the cluster solves.
2. **Requesting a GPU.** Show the diff between the CPU job script and the GPU job script. The key line: `#SBATCH --gres=gpu:1`. *Note:* keep CPU requests modest when requesting GPUs — you're not the only one on that node.
3. **Verifying the GPU is visible.** First line of any GPU script: `nvidia-smi`. Output goes to the SLURM log. *Pitfall:* PyTorch silently falls back to CPU if CUDA isn't available — the model trains, just 50× slower. Always check `torch.cuda.is_available()` in your script and fail loudly if it's False.
4. **GPU etiquette.** GPUs are the most contested resource on the cluster. Don't request a GPU you don't use. Don't hold an interactive GPU session while you go to lunch. Set tight time limits. *Note:* the queue for GPUs is longer than for CPUs — plan accordingly.
5. **Interactive GPU sessions for debugging.** `srun --gres=gpu:1 --pty bash`. Use this to iterate on model code interactively, then `sbatch` for the long training runs. *Note:* never debug via `sbatch` cycles — each one waits in the queue.
6. **The W9 deliverable.** Train the Fashion-MNIST CNN on a GPU on the cluster. Hand in: job script, training log showing GPU was used, final test accuracy. The transfer-learning bonus task should also use the GPU.

### Why this placement

By week 9, students have submitted multiple CPU jobs and are comfortable with the basic SLURM workflow. Adding GPU specifics now is incremental, not foundational. CNN training is the first task where the GPU is genuinely necessary, which makes the motivation self-evident.

This also gives students two full weeks (W9–W10) of GPU experience before the project, with the W10 transformer/LSTM exercise as a natural second GPU workout.

---

## Cumulative effect by Week 11 (Project Kickoff)

By the time projects begin, every student has:

- An SSH key registered, a working login, and a populated home directory on the cluster (since W1).
- Submitted at least one CPU batch job, debugged at least one queue/script issue, and retrieved results (since W4).
- Submitted at least one GPU job and verified GPU utilization (since W9).
- Had ~7 weeks of optional cluster use on exercises to internalize the workflow.

The project kickoff lecture (W11) then needs only a brief reminder ("you all know how to use the cluster — your project will run there") rather than a full infrastructure tutorial. This is exactly what we want: the project is about ML methodology, not about wrestling with new infrastructure during the highest-stakes phase of the course.

---

## A note on support load

The W1 touchpoint will generate a tail of "I can't log in" issues — possibly half the class in the first week. This is feature, not bug: surfacing problems now is the whole point. But plan for it:

- Pre-write a troubleshooting FAQ covering the most common issues (key format, public/private confusion, SSH config syntax, firewall/VPN requirements if relevant).
- Reserve a TA or office hours block specifically for cluster access in week 1.
- Have a fallback for students whose accounts genuinely don't work yet — they should still be able to do the W1 ML exercise locally and submit the cluster-check task late without penalty, as long as they engage with the access process.

By W4 the support load is mostly about SLURM scripts and resource requests, which is a more interesting (and more pedagogically useful) class of question.
