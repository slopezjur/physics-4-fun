# Mimic experiments use their isolated .venv, never the legacy MuJoCo training scripts.
# Paths below are relative to the project root unless absolute.
$Mimic = @{
    Python = 'mujoco_rig/mimic/.venv/Scripts/python.exe'
    MimicKit = $(if ($env:MIMICKIT_PATH) { $env:MIMICKIT_PATH } else { 'D:/Proyectos/Juegos/Tools/MimicKit' })
    Godot = 'D:/Programas/Godot_v4.7.1-stable_mono_win64/Godot_v4.7.1-stable_mono_win64.exe'
    LogRoot = 'logs/mimickit-training'
    DefaultExperiment = 'perturb'
    Minutes = 15
    Envs = 128
    Iterations = 10000
    Seed = 210921
    EvaluationInterval = 16
    ActorMaxKl = 0.02
    # A warm start loads weights + normalization, not optimizer/RNG state.
    InitializeFrom = 'logs/mimickit-stand/guarded-finetune-01/best.pt'
    SourceContract = 'logs/mimickit-stand/guarded-finetune-01/export/contract.json'
    WatchBundle = 'logs/mimickit-stand/guarded-finetune-01/export'
    Experiments = @{
        stand = @{ Task = 'stand' }
        perturb = @{
            Task = 'ball'
            BallSpeedMin = 1.0     # Horizontal m/s. Vertical speed compensates gravity at launch only.
            BallSpeedMax = 2.0
            QuietFraction = 0.25 # Keep undisturbed standing episodes during training.
        }
    }
}
# Ball mass/radius/friction/contact stiffness come from generated dummy_ball.xml.
# Do not edit generated physics to change difficulty: vary launch speed first.
