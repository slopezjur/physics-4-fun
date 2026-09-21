using System.Text.Json;
using Physics4Fun.RL.MuJoCo;
using Xunit;

namespace Physics4Fun.Tests;

public class MjMimicTargetControlTests
{
    private static JsonElement Contract(string feedforward = "none") => JsonSerializer.SerializeToElement(new
    {
        target_qpos_indices = Enumerable.Range(7, 30), target_qvel_indices = Enumerable.Range(6, 30),
        target_lower_radians = Enumerable.Repeat(-0.2f, 30), target_upper_radians = Enumerable.Repeat(0.2f, 30),
        residual_radians = 0.25f, pd_gain_per_torque_limit = 4f, pd_damping_time = 0.02f,
        target_timing = "sample_at_issue_hold_after_delay", initial_motor_control = "zero_until_valid_target",
        pd_update = "every_physics_step", feedforward
    });

    [Fact]
    public void DelayAppliesToTargetsWhileFeedbackUsesCurrentState()
    {
        var control = new MjMimicTargetControl(Contract());
        var q = new double[46]; var v = new double[45];
        var action = Enumerable.Repeat(0.4f, 30).ToArray();
        for (int i = 0; i < 2; i++)
        {
            control.Enqueue(q, v, action);
            Assert.Equal(0, control.Torque(0, q, v, 120));
        }
        control.Enqueue(q, v, new float[30]);
        Assert.Equal(48, control.Torque(0, q, v, 120), 4);
        q[7] = 0.15;
        Assert.True(control.Torque(0, q, v, 120) < 0);
        control.Reset();
        Assert.Equal(0, control.Torque(0, q, v, 120));
    }

    [Fact]
    public void PendingTargetsExposeClippedPositionVelocityAndValidityInOrder()
    {
        var control = new MjMimicTargetControl(Contract());
        var q = new double[46]; var v = new double[45]; v[6] = 0.1;
        control.Enqueue(q, v, Enumerable.Repeat(1f, 30).ToArray());
        var observation = new float[122]; int index = 0;
        control.Observe(observation, ref index);
        Assert.Equal(122, index);
        Assert.All(observation.Take(61), x => Assert.Equal(0, x));
        Assert.Equal(0.2f, observation[61]);
        Assert.Equal(0.1f, observation[91]);
        Assert.Equal(1, observation[121]);
    }

    [Fact]
    public void RejectsUnsupportedActuationAndWrongMotorMapping()
    {
        Assert.Throws<InvalidOperationException>(() => new MjMimicTargetControl(Contract("support_oracle")));
        var control = new MjMimicTargetControl(Contract());
        control.ValidateJointMapping(Enumerable.Range(1, 30).ToArray());
        Assert.Throws<InvalidOperationException>(() => control.ValidateJointMapping(Enumerable.Range(2, 30).ToArray()));
    }
}
