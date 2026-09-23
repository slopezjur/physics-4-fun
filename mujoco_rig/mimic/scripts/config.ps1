# Mimic experiments use their isolated .venv, never the legacy MuJoCo training scripts.
# Paths below are relative to the project root unless absolute.
$Mimic = @{
    Python = 'mujoco_rig/mimic/.venv/Scripts/python.exe'
    MimicKit = 'tools/MimicKit'
    Godot = 'tools/Godot/Godot.exe'
    LogRoot = 'logs/mimickit-training'
    DefaultExperiment = 'perturb'
    Minutes = 30
    Envs = 2048
    Iterations = 1000000
    Seed = 210921
    # Full 96-case native validation every N completed PPO updates.
    # Initial and final validation always run (or reuse identical cached results).
    EvaluationInterval = 64
    ActorMaxKl = 0.03
    # Defaults preserve the current guarded target-PD fine-tuning path. Override them per run
    # with train.ps1 -Control/-Stability without changing this file.
    Control = 'target_pd'
    # Legacy checkpoints require legacy semantics. Use support_v2 with a matching
    # checkpoint, or start a fresh Stand experiment with -Stability upstream.
    ContactMode = 'legacy'
    Stability = 'guarded'
    # A warm start loads weights + normalization, not optimizer/RNG state.
    InitializeFrom = 'logs/mimickit-perturb/guarded-perturb-04/best.pt'
    SourceContract = 'logs/mimickit-perturb/guarded-perturb-04/export/contract.json'
    WatchBundle = 'logs/mimickit-perturb/guarded-perturb-04/export'
    Experiments = @{
        stand = @{ Task = 'stand' }
        perturb = @{
            Task = 'ball'
            BallSpeedMin = 1.0     # Horizontal m/s. Vertical speed compensates gravity at launch only.
            BallSpeedMax = 2.5
            QuietFraction = 0.25 # Keep undisturbed standing episodes during training.
            BallReward = 'reference' # Opt in per run with -BallReward recovery_v1 until validated.
        }
    }
}
# Ball mass/radius/friction/contact stiffness come from generated dummy_ball.xml.
# Do not edit generated physics to change difficulty: vary launch speed first.

# Keep machine-specific installations out of version control. Environment overrides win.
$localConfig = Join-Path $PSScriptRoot 'config.local.ps1'
if (Test-Path -LiteralPath $localConfig) { . $localConfig }
if ($env:MIMICKIT_PATH) { $Mimic.MimicKit = $env:MIMICKIT_PATH }
if ($env:P4F_GODOT_EXE) { $Mimic.Godot = $env:P4F_GODOT_EXE }
