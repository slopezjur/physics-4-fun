using System;

namespace Physics4Fun.Ragdoll.Trajectories;

internal sealed class PushUpDrillClock
{
    private const float SetupSeconds = 1.0f;
    private const float RepSeconds = 2.4f;
    private float _elapsedSeconds;

    public float CycleNormalized
    {
        get
        {
            if (_elapsedSeconds < SetupSeconds)
            {
                return 0.0f;
            }

            float phase = (_elapsedSeconds - SetupSeconds) / RepSeconds;
            return 0.5f * (1.0f - MathF.Cos(phase * 2.0f * MathF.PI));
        }
    }

    public void Advance(float deltaSeconds) => _elapsedSeconds += deltaSeconds;

    public void Reset() => _elapsedSeconds = 0.0f;
}
