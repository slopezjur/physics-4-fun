using System.Collections.Generic;
using Godot;
using Physics4Fun.RL.Interfaces;

namespace Physics4Fun.RL.Curriculum;

/// <summary>
/// Reverse curriculum over the get-up start pose, expressed as a floor on the pose parameter t.
///
/// t = 1.0 is fully standing and t = 0.0 is flat on the floor, so the curriculum walks its floor
/// DOWN over time: the policy first learns to stand from almost-standing, then from progressively
/// worse poses, each level bootstrapped by the one before it. Starting at the hard end instead
/// (t = 0, prone) is the configuration that produced 0.000 success across 1179 rollouts, because
/// the reward gives nothing until a full get-up happens by chance and it never does.
///
/// Extracted from RagdollRLBridge, where it was 83 lines of independent state and policy sharing a
/// class with episode plumbing, observation marshalling and provenance reporting. It has its own
/// state, its own persistence across resumes and its own advance rule, none of which the bridge
/// needs to know about - and this project already keeps reward, termination, observation and action
/// behind interfaces for the same reason.
/// </summary>
public sealed class ReverseStartPoseCurriculum : IRlStartPoseCurriculum
{
    private readonly float _step;
    private readonly float _frontierShare;
    private readonly int _window;
    private readonly float _advanceRate;

    /// <summary>Trailing outcomes at the current level; cleared, not slid, on advance.</summary>
    private readonly Queue<bool> _history = new();

    public ReverseStartPoseCurriculum(float startFloor, float step, float frontierShare, int window, float advanceRate)
    {
        _step = Mathf.Max(0.0001f, step);
        _frontierShare = Mathf.Clamp(frontierShare, 0.0f, 1.0f);
        _window = Mathf.Max(1, window);
        _advanceRate = Mathf.Clamp(advanceRate, 0.0f, 1.0f);
        Floor = Mathf.Clamp(startFloor, 0.0f, 1.0f - _step);
    }

    /// <summary>Lowest (hardest) start pose the curriculum has unlocked. Decreases over a run.</summary>
    public float Floor { get; private set; }

    /// <summary>
    /// Restores a floor recovered from a previous run's TensorBoard history.
    ///
    /// This is the whole reason the floor is not simply re-read from the scene on every start. A
    /// resumed run that reverts to the scene default throws away every level the curriculum has
    /// cleared, and it does so silently: measured on getup_v4_2, standing/curriculum_t reported
    /// min 0.9080 and last 0.9900 - the floor reached 0.908 and was reset on every restart, so
    /// months of chained sessions all relearned the same first step.
    /// </summary>
    public void RestoreFloor(float floor)
    {
        Floor = Mathf.Clamp(floor, 0.0f, 1.0f - _step);
        _history.Clear();
    }

    /// <summary>
    /// Draws a start pose from the curriculum's own range. Standing-refresher episodes are the
    /// caller's business, not the curriculum's - it only knows about poses it has unlocked.
    /// </summary>
    public float SampleStartPose()
    {
        // Frontier band: the one step the curriculum is actually trying to clear. Only these vote
        // (see IsFrontierPose), so their share has to be held fixed rather than left to shrink as
        // the range widens.
        if (GD.Randf() < _frontierShare)
        {
            return Mathf.Lerp(Floor, Mathf.Min(1.0f, Floor + _step), GD.Randf());
        }

        // Rehearsal over everything already cleared, so competence at earlier levels is not traded
        // away while the frontier is being learned.
        return Mathf.Lerp(Floor, 1.0f, GD.Randf());
    }

    /// <summary>
    /// Whether a start pose sits in the band the curriculum is currently trying to clear, and so
    /// counts toward the advance decision. Rehearsal and standing-refresher episodes are excluded:
    /// they are drawn from levels already passed, so letting them vote measures the past.
    /// </summary>
    public bool IsFrontierPose(float poseT) => poseT < 1.0f && poseT <= Floor + _step;

    /// <summary>
    /// Feeds one frontier episode's outcome to the trailing window and advances the level when the
    /// window clears the advance rate.
    ///
    /// The window is CLEARED on advance, not slid. A slid window still holds the successes that
    /// triggered the promotion, so it would re-trigger within an episode or two and run the level
    /// down several steps before the policy has seen the new distribution at all. Clearing forces a
    /// full window of fresh evidence before the next move.
    ///
    /// Returns what happened rather than logging it, so the decision stays testable without a
    /// scene tree and the caller keeps ownership of what reaches the console.
    /// </summary>
    public CurriculumAdvance RecordOutcome(bool succeeded)
    {
        if (Floor <= 0.0f)
        {
            return CurriculumAdvance.None;
        }

        _history.Enqueue(succeeded);
        while (_history.Count > _window)
        {
            _history.Dequeue();
        }

        if (_history.Count < _window)
        {
            return CurriculumAdvance.None;
        }

        int wins = 0;
        foreach (bool outcome in _history)
        {
            if (outcome) { wins++; }
        }

        int considered = _history.Count;
        if ((float)wins / considered < _advanceRate)
        {
            return CurriculumAdvance.None;
        }

        float previous = Floor;
        Floor = Mathf.Max(0.0f, Floor - _step);
        _history.Clear();

        return new CurriculumAdvance(true, previous, Floor, wins, considered);
    }

    public string Describe() =>
        $"ReverseStartPoseCurriculum(floor={Floor:F3}, step={_step}, frontierShare={_frontierShare}, "
        + $"window={_window}, advanceRate={_advanceRate})";
}
