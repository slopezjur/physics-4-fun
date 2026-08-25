using Godot;

namespace Physics4Fun.Ragdoll;

/// <summary>
/// Evaluates pelvis/body coordinate frames relative to world gravity to determine anatomical
/// orientation.
///
/// <para><b>Per-ragdoll, deliberately not static.</b> The classifier is a hysteresis latch: every
/// branch below reads AND writes <see cref="_previous"/>, so the instance carries the body's
/// recent history rather than being a pure function of the pose it is handed.</para>
///
/// <para>It used to be a static class with a static <c>_previous</c>, which was correct only while
/// a process simulated exactly one ragdoll. <see cref="RL.RagdollSpawner"/> now puts up to 64 in
/// one process, and <c>BalanceController.EvaluateState</c> runs for every one of them on every
/// physics tick - including in the RL state, where it is not gated off. All 64 bodies therefore
/// shared a single latch, each overwriting it for whichever body was classified next, so the
/// hysteresis did not merely stop working: one body's orientation decided another body's
/// branch.</para>
///
/// <para>That inverts the entire point of the hysteresis. Its own rationale below is that a
/// flickering orientation "thrashes bone targets between incompatible poses and the ragdoll never
/// rises" - and a shared latch produces exactly that flicker, on the get-up task, which is the
/// project's stated unsolved goal.</para>
/// </summary>
public sealed class OrientationClassifier
{
    // Hysteresis state: enter Upright when upDot > 0.70, remain Upright until upDot < 0.45
    private RagdollOrientation _previous = RagdollOrientation.Upright;

    /// <summary>
    /// The orientation this body is in, latched against its own previous answer.
    ///
    /// Call once per tick per body. Calling it twice in a tick is not a read-only query - the
    /// second call sees the latch the first one set.
    /// </summary>
    public RagdollOrientation Classify(Basis pelvisBasis)
    {
        Vector3 pelvisUp = pelvisBasis.Y.Normalized();
        Vector3 pelvisForward = -pelvisBasis.Z.Normalized();

        float upDot = pelvisUp.Dot(Vector3.Up);
        float forwardDot = pelvisForward.Dot(Vector3.Up);

        if (upDot > 0.70f || (_previous == RagdollOrientation.Upright && upDot > 0.45f))
        {
            _previous = RagdollOrientation.Upright;
            return RagdollOrientation.Upright;
        }

        // Hysteresis mirrors the Upright check above: entering Supine/Prone needs a strong
        // signal (|forwardDot| > 0.20), but once classified, small wobbles near the boundary
        // don't immediately kick it back out to Side. Without this, a wobbling pelvis during
        // recovery flickers Prone<->Side<->Supine every tick; since BiomechanicalMotionSynthesizer
        // dispatches a different trajectory per orientation, the flicker thrashes bone targets
        // between incompatible poses and the ragdoll never rises.
        if (forwardDot > 0.20f || (_previous == RagdollOrientation.Supine && forwardDot > 0.05f))
        {
            _previous = RagdollOrientation.Supine;
            return RagdollOrientation.Supine; // Chest facing sky (on back)
        }

        if (forwardDot < -0.20f || (_previous == RagdollOrientation.Prone && forwardDot < -0.05f))
        {
            _previous = RagdollOrientation.Prone;
            return RagdollOrientation.Prone;  // Chest facing ground (on belly)
        }

        _previous = RagdollOrientation.Side;
        return RagdollOrientation.Side;
    }

    /// <summary>
    /// Clears the latch back to Upright.
    ///
    /// Call when a body is teleported or respawned: the latch describes continuous motion, and
    /// carrying a pre-reset orientation across a discontinuity is the one case where hysteresis is
    /// actively wrong.
    /// </summary>
    public void Reset() => _previous = RagdollOrientation.Upright;
}
