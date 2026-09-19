# ============================================================================
#  EDIT THIS FILE to change how MuJoCo training runs. Every script reads these values.
#
#  The MuJoCo track differs from the Isaac track in one way that governs everything below:
#  **the engine that trains is the engine that ships.** Godot drives MuJoCo's C library directly
#  through P/Invoke (Source/RL/MuJoCo/MjBridge.cs), so there is no sim-to-sim boundary to lose a
#  policy across - which is what defeated every Isaac attempt.
#
#  Moving TRAINING to the GPU reopens that boundary a crack, because mujoco_warp is float32 and
#  the C engine is float64. The rule that keeps it closed is not negotiable:
#
#      Train on GPU. Score and ship on CPU MuJoCo.
#      A checkpoint is never judged by the engine that trained it.
#
#  $EvalBackend below is therefore 'cpu' and should stay that way.
# ============================================================================

# --- Paths -------------------------------------------------------------------
# The conda environment holding mujoco, mujoco_warp, warp and torch+cu128. Named, not pathed.
#
# This is env_isaaclab3, shared with the Isaac Lab 3 track - mujoco_warp arrived there as a Newton
# dependency, which is why the GPU backend needed no new install. Do NOT disturb the older
# env_isaaclab; the Isaac 2 track still uses it.
$CondaEnvName = 'env_isaaclab3'

# Resolved rather than assumed, because a bare `python` on a machine like this is typically a
# standalone interpreter with none of the above, and the only symptom is "ModuleNotFoundError: No
# module named 'mujoco'", which says nothing about the real cause.
#
# **Nothing here hardcodes a personal install path.** The env is found BY NAME, by asking conda
# where it lives - $env:CONDA_EXE and $env:CONDA_PREFIX are set inside any conda shell, and
# `conda info --base` answers outside one. (isaac_lab/scripts/config.ps1 and
# isaac_lab_3/scripts/config.ps1 both hardcode 'D:\Programas\anaconda3\...' as their second
# resolution step; this is the same idea without the machine-specific literal. A conda env cannot
# be a path relative to the repo - it lives outside it - so "by name, ask the tool" is the portable
# equivalent.)
#
# Resolution order:
#   1. $env:P4F_MUJOCO_PYTHON - an explicit override, for a non-conda or unusually placed install
#   2. the conda root reported by CONDA_EXE / CONDA_PREFIX / `conda info --base`
#   3. envs listed by `conda env list`
#   4. the conventional roots under the user profile and ProgramData
function Resolve-MujocoPython {
    param([string] $EnvName)

    if ($env:P4F_MUJOCO_PYTHON) {
        if (Test-Path $env:P4F_MUJOCO_PYTHON) { return (Resolve-Path $env:P4F_MUJOCO_PYTHON).Path }
        throw "P4F_MUJOCO_PYTHON is set to '$env:P4F_MUJOCO_PYTHON' but no file exists there."
    }

    # conda records every environment prefix it creates in this file. It is the canonical answer to
    # "where are the envs on THIS machine", it needs neither conda on PATH nor an activated shell,
    # and it lives in the user profile rather than in the repo - which is the whole point. On this
    # machine conda is installed to a non-standard drive and is not on PATH in a non-interactive
    # shell, so without this step nothing else finds it.
    $registry = Join-Path $env:USERPROFILE '.conda\environments.txt'
    if (Test-Path $registry) {
        foreach ($line in (Get-Content $registry -ErrorAction SilentlyContinue)) {
            $prefix = $line.Trim()
            if (-not $prefix -or -not (Test-Path $prefix)) { continue }
            # The file lists both base installs and envs, so try the prefix as the env itself and
            # as a root containing envs\<name>.
            if ((Split-Path $prefix -Leaf) -eq $EnvName) {
                $candidate = Join-Path $prefix 'python.exe'
                if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
            }
            $candidate = Join-Path $prefix "envs\$EnvName\python.exe"
            if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
        }
    }

    $roots = New-Object System.Collections.Generic.List[string]
    # Inside an activated conda shell both of these are set; CONDA_PREFIX may be the env itself,
    # in which case its grandparent is the root.
    if ($env:CONDA_EXE) { $roots.Add((Split-Path (Split-Path $env:CONDA_EXE -Parent) -Parent)) }
    if ($env:CONDA_PREFIX) {
        $roots.Add($env:CONDA_PREFIX)
        $parent = Split-Path $env:CONDA_PREFIX -Parent
        if ((Split-Path $parent -Leaf) -eq 'envs') { $roots.Add((Split-Path $parent -Parent)) }
    }
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        try {
            $base = (& conda info --base 2>$null | Select-Object -First 1)
            if ($base) { $roots.Add($base.Trim()) }
        } catch { }
    }
    $roots.Add("$env:USERPROFILE\anaconda3")
    $roots.Add("$env:USERPROFILE\miniconda3")
    $roots.Add("$env:LOCALAPPDATA\anaconda3")
    $roots.Add('C:\ProgramData\anaconda3')
    $roots.Add('C:\ProgramData\miniconda3')

    foreach ($root in $roots) {
        if (-not $root) { continue }
        $candidate = Join-Path $root "envs\$EnvName\python.exe"
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }

    # Last resort: ask conda to enumerate. Catches envs created with --prefix somewhere unusual.
    if ($conda) {
        try {
            foreach ($line in (& conda env list 2>$null)) {
                if ($line -match '^\s*#') { continue }
                $path = ($line -split '\s+' | Where-Object { $_ -match '[\\/]' } | Select-Object -Last 1)
                if ($path -and (Split-Path $path -Leaf) -eq $EnvName) {
                    $candidate = Join-Path $path 'python.exe'
                    if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
                }
            }
        } catch { }
    }

    throw @"
Could not locate the conda environment '$EnvName'.

This track needs mujoco, mujoco_warp and a CUDA torch. Either create/activate that environment, or
point P4F_MUJOCO_PYTHON at an interpreter that has them:

    setx P4F_MUJOCO_PYTHON "<path to>\python.exe"

then open a NEW terminal (setx only affects processes started afterwards).
"@
}

$Python = Resolve-MujocoPython -EnvName $CondaEnvName

# Cheap sanity check on the resolved interpreter - filesystem only, so it costs nothing. Catches
# "resolved SOME python, but not the one with the packages", which otherwise surfaces much later as
# a ModuleNotFoundError from inside a training run.
$sitePackages = Join-Path (Split-Path $Python -Parent) 'Lib\site-packages'
foreach ($pkg in @('mujoco', 'mujoco_warp', 'warp', 'torch')) {
    if (-not (Test-Path (Join-Path $sitePackages $pkg))) {
        Write-Host "[config] WARNING: '$pkg' not found under $sitePackages." -ForegroundColor Yellow
        Write-Host "         Resolved Python: $Python" -ForegroundColor Yellow
        Write-Host "         Set P4F_MUJOCO_PYTHON if this is the wrong interpreter." -ForegroundColor Yellow
    }
}

# --- Which task to train -----------------------------------------------------
#   "perturb" - stay standing through ball impacts, using the legs. No command channel.
#   "walk"    - follow a commanded forward speed, lateral speed and turn rate.
#
# They share the body, the actuators and the plumbing; they differ in the observation tail (walk
# adds three command channels, 120 -> 123) and in the reward. Scripts accept -Task to override this
# for one command without editing the file.
$Task = "perturb"

# Commanded velocity used when SCORING a walk policy and when driving the Godot scene:
# forward m/s, lateral m/s, turn rate rad/s.
$WalkCommand = @(0.6, 0.0, 0.0)

# --- Backend -----------------------------------------------------------------
#   "warp" - mujoco_warp on the GPU. Thousands of worlds in one batched CUDA kernel.
#   "cpu"  - MuJoCo's C engine, one MjData per env stepped in a Python loop.
#
# THROUGHPUT, measured on this machine (RTX 4080 SUPER 16 GB, Ryzen 7 7800X3D 8c/16t) by
# scripts/bench.py: 30 configurations, 45 s of timed work each, one subprocess per configuration,
# resources sampled while in flight. Samples/s, by envs (rows) and rollout steps (columns):
#
#     envs      steps 4   steps 8  steps 16  steps 24 |  GPU%     VRAM
#     cpu 96          -     2,761         -     3,086 |   2%      0.5G
#      2,048     22,896    26,850    28,922    29,210 |  51%      1.3G
#      4,096     36,838    41,648    41,059    44,228 |  64%      2.1G
#      8,192     55,919    59,731    60,898    60,637 |  79%      2.9G
#     12,288     67,350    71,505    72,293    71,966 |  82%      3.7G
#     16,384     71,326    76,271    77,626    77,990 |  95%      4.7G
#     24,576     78,145    81,388    81,965    81,802 |  97%      6.5G
#     32,768     80,637    83,028    82,066    83,448 |  97%      8.4G
#
# Throughput plateaus around 24,576 at ~83,000 samples/s, 30x the CPU. Memory was never the
# constraint on THAT plant - the largest configuration peaked at 8.4 GiB of 16.
#
# **Those numbers are the POSITION plant and no longer apply.** A body that actually crumples
# generates real constraints, so both the solver cost and the constraint buffers grew (NJMAX 128 ->
# 256). Re-measured on the torque plant, 1 minute per configuration, one at a time:
#
#     config          batch      it/min   policy steps/s   peak VRAM   GPU
#     16384 x 16     262,144       6.25         27,307     10,698 MiB   65%   <- in use
#     32768 x 8      262,144       5.00         21,845     15,907 MiB   65%
#     65536 x 4      262,144       1.43          6,242     16,023 MiB   18%   VRAM WALL
#     24576 x 16     393,216       4.17         27,307     12,690 MiB   69%
#     49152 x 8      393,216       1.43          9,362     15,999 MiB   44%   VRAM WALL
#     32768 x 16     524,288       3.33         29,127     15,891 MiB   69%
#
# Sustained over a full 15-minute run, 16,384 x 16 delivers ~21,800 policy steps/s - 44% below the
# position plant, and the difference IS the physics being real.
#
# **VRAM is now the binding constraint, not compute.** The two collapsed rows sat at 16.0 GiB, the
# card's ceiling; their low utilisation is allocator thrashing, not a slower configuration. The
# dense constraint Jacobian is nworld x njmax x nv floats, which at nv=51 is 0.80 GiB for 16,384
# worlds and 1.59 GiB for 32,768. Nothing above 16,384 envs has headroom, and 32,768 x 16 buys 7%
# throughput - inside the noise of a 1-minute sample - at 99% VRAM.
#
# Utilisation 58-95% at 92-116 W of a 320 W card says the GPU is occupancy-bound, not compute-bound.
#
# The CPU path is single-threaded regardless of core count: perturb_env.py steps every MjData in a
# Python for-loop, so it uses one core of eight and the 4080 sits at 0%. The GPU path uses only
# 6-8% of the CPU, so the host is not the limit either - an attempt to remove the remaining host
# syncs from the env made it 9-16% SLOWER (see _fire_ball in perturb_env_warp.py).
#
# **The two engines are not interchangeable.** mujoco_warp is float32, the C engine float64, and
# `parity_gpu.py` measures the consequence: identical open-loop control diverges to 0.14 degrees of
# pelvis tilt by 2.5 s. A CPU-vs-CPU control run started 1 nanometre apart does NOT diverge at all,
# so this is genuinely the engine and not chaos. Train on it; never score on it.
$Backend = "warp"

# Backend used by eval.ps1. **Leave this on 'cpu'** - it is the engine Godot ships.
$EvalBackend = "cpu"

# --- Training scale ----------------------------------------------------------
# **NOT the row with the best samples/s.** 32,768 x 24 tops the throughput table at 83,448
# samples/s and is a bad setting: it updates the policy once every nine seconds. What trains the
# dummy is the number of POLICY UPDATES and their quality - samples/s is an input, not a result.
#
# LEARNING, measured directly: identical task, difficulty pinned, same seed, reading the episode
# length reached of 1199. This is the measurement that decides $Envs and $Steps.
#
# **RE-MEASURED on the TORQUE plant, 2026-09-10** - `mujoco_rig/scripts/batch_table.py`, walk task,
# same seed, 10 minutes per cell, one trainer at a time. This replaces the table below it, which was
# measured on the position-actuated body and does not apply.
#
#   envs x steps      batch    it/min   updates/h   peak ep_len   end   retained    vx
#    16384 x 4       65,536     21.0       1,260           373     338      91%    -0.05
#    16384 x 8      131,072     10.3         616           426     413      97%    +0.01
#    16384 x 16     262,144      5.1         307           504     399      79%    +0.01
#     4096 x 16      65,536     18.0       1,080         1,196     994      83%    +0.01   <- BEST
#
# **Two findings, and both overturn what this file used to say.**
#
#   1. **65,536 does not collapse any more.** The old threshold - "below ~65k PPO climbs then
#      crashes to the floor" - was a property of the POSITION plant. On the torque plant 16,384x4
#      retains 91% and 16,384x8 retains 97%, both better than the 262,144 batch that was being run
#      out of caution. Respecting a stale threshold cost 4x the wall-clock per policy update.
#   2. **The SPLIT matters more than the batch, again.** The two 65,536 cells are not
#      interchangeable: 4,096x16 peaked at 1,196 episode steps against 16,384x4's 373, at a similar
#      update rate. Four rollout steps is 67 ms of experience per update - too little for the value
#      function to see the consequence of a step before the batch ends. Rollout LENGTH is doing
#      something batch size does not capture.
#
# So the default is 4,096 x 16, and `train.py --envs_schedule` grows the world count while holding
# the rollout at 16. Caveat, stated plainly: one cell each, ten minutes, walk only. The 3.2x gap
# between the two 65,536 cells is far too large to be noise, but the middle rows are not settled
# and none of this is validated for perturb.
#
# --- the older table, POSITION plant, kept for the reasoning below it -----------------------
# **Measured on THIS machine, on the compliant POSITION plant, one trainer at a time.** Identical
# task, identical seed checkpoint, identical 5 minutes each; the only difference between rows is
# the envs x steps split. `mujoco_rig/scripts/batch_table.py` reproduces it.
#
#   envs x steps      batch   iters  peak ep  end ep  retained  verdict
#    16384 x 4       65,536     320     1109     115       10%  COLLAPSED
#     8192 x 8       65,536     275     1199     110        9%  COLLAPSED
#    32768 x 4      131,072     185      507     498       98%
#    16384 x 8      131,072     170     1199       80       7%  COLLAPSED
#    24576 x 8      196,608     125      606     594       98%
#    32768 x 8      262,144      95      586     527       90%
#    16384 x 16     262,144      85     1192    1006       84%   <- BEST
#    24576 x 16     393,216      60      788     614       78%
#
# **Read the RETAINED column, never the peak.** end/peak is what separates a policy that learned
# something from one that reached a number and then lost it. Every collapsed row here has a fine
# peak - 8,192x8 touched the 1199 ceiling and finished at 110.
#
# Three conclusions:
#
#   1. **65,536 is not enough on this plant.** Both 65,536 rows collapsed. The threshold moved:
#      that batch was stable on the older, over-stiff body, and the compliant one is a harder task
#      needing a cleaner gradient. A stability threshold does not survive a change of plant.
#   2. **The batch is what matters, not the split - mostly.** At 262,144 the two splits both
#      survive. But 131,072 splits DISAGREE (32,768x4 keeps 98%, 16,384x8 keeps 7%), so the split
#      is not entirely irrelevant near the threshold, and a single sample per cell cannot say more
#      than that.
#   3. **Highest retained is not the goal either.** 32,768x4 keeps 98% of a peak of 507; 16,384x16
#      keeps 84% of 1192 and ends at 1006 of a possible 1199. The high-retention rows are stable
#      because they barely moved.
# 4,096 - not 16,384. See the torque-plant table above: same batch as 16,384x4 at a similar update
# rate, and 3.2x the peak episode length. `--envs_schedule 4096,8192,16384` grows it during a run.
$Envs = 4096

# Rollout steps per env per update. $Envs x $Steps is the BATCH, and the batch is what the table
# above is really about: 16,384 x 16 = 262,144 samples per update.
#
# Do not lower this without lowering $Envs to match, and do not raise $Envs without checking the
# batch stays above the collapse threshold. The two knobs are not independent - only their product
# matters for stability, and their ratio only for how many updates you get per second.
$Steps = 16

# 0 = run until $MaxMinutes.
$Iterations = 0

# Wall-clock cap in minutes; training stops at whichever comes first. The run always stops on a
# checkpoint boundary.
$MaxMinutes = 30

# --- PPO ---------------------------------------------------------------------
# Initial policy std. **Measured against the plant, not inherited from a tutorial.**
# probe_noise.py: what governs survival is std x $Authority - the size of the random jump in the
# joint target, which a position actuator at kp up to 1800 answers immediately.
#
#     std x authority   0.0375  0.0250  0.0125  0.0100  0.0075  0.0050
#     time to fall       1.57s   2.33s   6.96s  10.81s   never   never
#
# At the 0.25 authority a protective step needs, 0.03 is the largest std with a 100% survival rate.
# The textbook 0.5 puts the body on the floor in 1.6 s - before the first ball even fires.
$InitStd = 0.03

# Summed over 36 action dims the usual 0.005 outweighs the advantage signal and drives std UP
# (0.500 -> 0.512 over 475 iterations) on a plant where randomness is what topples the body.
$EntropyCoef = 0.0005

# KL target for the adaptive learning rate. A GPU batch has a far cleaner gradient than a CPU one,
# so it can afford a larger step; this is where the extra samples get spent.
$DesiredKl = 0.01
$Epochs = 5

# --- The task ----------------------------------------------------------------
# Episode length in seconds. The unaided body's median fall under fire is 12.8 s, so a 12 s episode
# leaves almost no headroom to measure improvement against - 20 s does.
$Seconds = 20.0

# Seconds between ball impacts, min and max.
#
# **The authored 2-4 s made the task impossible for every controller.** A single 10 kg impact is
# survivable 75% of the time with no policy at all, but the same ball every 2-4 s topples 8/8 in
# 6.8 s, because the next one lands mid-recovery and no recovery ever finishes.
$BallEvery = @(4.0, 7.0)

# --- Curriculum --------------------------------------------------------------
# Impulse is m*dv, so ball SPEED ramps difficulty with no model recompile - which matters, because
# rewriting `body_mass` on a live MjModel leaves the contact solver's invweight constants stale and
# produced a sweep saying a 0.5 kg ball was more destructive than a 10 kg one.
#
# A protective step is too large a behaviour for 0.03 std to discover, so it has to be GROWN: every
# increment stays inside the small noise ball around the policy that already works.
$SpeedStart = 3.0      # 30 N.s on the 10 kg ball - about where the unaided body starts failing
$SpeedEnd   = 6.0      # what the Godot Perturb scene actually fires
$SpeedStep  = 1.10

# Promotion needs near-perfect survival AND a dwell. The first curriculum promoted on "75% of an
# episode survived" with no dwell, went 3.0 -> 6.0 m/s in 2.2 minutes, and episode length fell
# straight back to the plateau the curriculum existed to escape.
$PromoteAt = 0.93

# Dwell is counted in EPISODES, not iterations - a GPU iteration carries ~28x the experience of a
# CPU one, so an iteration count would mean something different on each backend.
$StageMinEpisodes = 400

# --- Evaluation --------------------------------------------------------------
# Scored on CPU MuJoCo, always. More envs is a tighter measurement and nothing else.
$EvalEnvs = 12

# 40 seconds, not 20. A 20 s window passed a body that was already toppling three separate times on
# this project; travel is also reported per quarter, because a fall produces distance.
$EvalSeconds = 40.0

# The number to beat, measured with NO policy at $BallEvery and $SpeedEnd:
#     under fire   30.1% upright, 0/12 survived, median fall 12.8 s
#     quiet room  100.0% upright, 12/12 survived
# The quiet row matters: an all-zero action is a stiff PD hold on the rest pose and it stands
# indefinitely, so any policy scoring below 100% quiet has DESTROYED a working controller.

# --- Run selection -----------------------------------------------------------
# Scripts pick the newest run that reached at least this many checkpoints. "Newest run" alone is
# wrong: a throwaway smoke run finishing later than the real one makes every script silently use
# it, and the numbers look plausible.
$MinIterations = 25

# Pin a specific run directory instead of taking the newest, e.g. "2026-09-08_23-06-28_gpu_smoke".
$RunName = ""

# ============================================================================
#  Derived - normally no need to edit below here.
# ============================================================================
$MujocoRoot  = (Resolve-Path "$PSScriptRoot/..").Path
$ProjectRoot = (Resolve-Path "$PSScriptRoot/../..").Path
$RlDir       = Join-Path $MujocoRoot "rl"
$LogRoot     = Join-Path $ProjectRoot "logs\mujoco"

# A script may set these from its own parameters before dot-sourcing this file.
if ($BackendOverride) { $Backend = $BackendOverride }
if ($EnvsOverride -gt 0) { $Envs = $EnvsOverride }
if ($MinutesOverride -ge 0) { $MaxMinutes = $MinutesOverride }

if ($TaskOverride) { $Task = $TaskOverride }
if ($Task -ne "perturb" -and $Task -ne "walk") {
    throw "Unknown `$Task '$Task'. Use 'perturb' or 'walk'."
}
if ($Backend -ne "warp" -and $Backend -ne "cpu") {
    throw "Unknown `$Backend '$Backend'. Use 'warp' (GPU) or 'cpu'."
}
if ($EvalBackend -ne "cpu") {
    Write-Host "[config] WARNING: `$EvalBackend is '$EvalBackend', not 'cpu'. Scoring on the GPU" -ForegroundColor Yellow
    Write-Host "         engine measures a policy against float32 physics that Godot will never" -ForegroundColor Yellow
    Write-Host "         run. This is the exact mistake the Isaac track never recovered from." -ForegroundColor Yellow
}
if ($Envs -lt 1) { throw "`$Envs must be at least 1 (got $Envs)." }
if ($Steps -lt 1) { throw "`$Steps must be at least 1 (got $Steps)." }
if ($SpeedEnd -lt $SpeedStart) { throw "`$SpeedEnd ($SpeedEnd) is below `$SpeedStart ($SpeedStart)." }
if ($Backend -eq "cpu" -and $Envs -gt 512) {
    Write-Host "[config] NOTE: $Envs envs on the CPU backend is a single-threaded Python loop." -ForegroundColor Yellow
    Write-Host "         Measured 2,693 policy SPS at 96 envs; this will be slow, not parallel." -ForegroundColor Yellow
}

# torch.onnx.export prints a U+2705 on success, which a default Windows console (cp1252) cannot
# encode - the export then dies in the logging call, AFTER the graph is already built.
$env:PYTHONIOENCODING = "utf-8"

# Newest run directory holding at least $Minimum checkpoints. Returns $null rather than throwing so
# callers can report "not trained yet" in their own words.
function Get-MujocoRun {
    param(
        [int] $Minimum = 25,
        [string] $Pinned = ""
    )
    if (-not (Test-Path -LiteralPath $LogRoot)) { return $null }

    if ($Pinned) {
        $dir = Join-Path $LogRoot $Pinned
        if (-not (Test-Path -LiteralPath $dir)) { throw "No run '$Pinned' under $LogRoot." }
        return (Get-Item -LiteralPath $dir)
    }

    foreach ($dir in (Get-ChildItem -LiteralPath $LogRoot -Directory -ErrorAction SilentlyContinue |
                      Sort-Object LastWriteTime -Descending)) {
        $ckpts = @(Get-ChildItem -LiteralPath $dir.FullName -Filter 'model_*.pt' -ErrorAction SilentlyContinue)
        if ($ckpts.Count -ge 1) {
            $best = ($ckpts | ForEach-Object { [int]($_.BaseName -replace '^model_', '') } |
                     Measure-Object -Maximum).Maximum
            if ($best -ge $Minimum) { return $dir }
        }
    }
    return $null
}

# Highest-numbered checkpoint in a run. Sorting by name puts model_9 after model_100.
function Get-MujocoCheckpoint {
    param([Parameter(Mandatory = $true)][string] $RunDir)
    $ckpts = @(Get-ChildItem -LiteralPath $RunDir -Filter 'model_*.pt' -ErrorAction SilentlyContinue)
    if ($ckpts.Count -eq 0) { throw "No model_*.pt in $RunDir." }
    return ($ckpts | Sort-Object { [int]($_.BaseName -replace '^model_', '') } -Descending |
            Select-Object -First 1).FullName
}

# Runs a script from mujoco_rig/rl with the resolved Python, from the PROJECT ROOT - the scripts
# resolve logs/ and mujoco_rig/ relative to the working directory.
function Invoke-Mujoco {
    param(
        [Parameter(Mandatory = $true)][string] $Script,
        [string[]] $Arguments = @()
    )
    Push-Location $ProjectRoot
    try {
        & $Python -u (Join-Path $RlDir $Script) @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Script exited with code $LASTEXITCODE." }
    } finally {
        Pop-Location
    }
}

$where = if ($Backend -eq "warp") { "GPU (mujoco_warp, float32)" } else { "CPU (MuJoCo C, float64)" }
Write-Host "[config] task '$Task' on $where, $Envs envs x $Steps steps; score on $EvalBackend." -ForegroundColor DarkGray
