using System.Text.Json;
using Physics4Fun.RL.MuJoCo;
using Xunit;

public sealed class MjMimicContactContractTests
{
    [Theory]
    [InlineData("mimic_stand_v1")]
    [InlineData("mimic_stand_target_pd_v1")]
    public void LegacyKeepsItsSemantics(string schema)
    {
        using var json = JsonDocument.Parse(JsonSerializer.Serialize(new { observation_contract = schema }));
        Assert.False(MjMimicContactContract.IsCorrected(json.RootElement));
    }

    [Fact]
    public void CorrectedRequiresSamplingAndOrderedSupportBodies()
    {
        using var json = JsonDocument.Parse(JsonSerializer.Serialize(new {
            observation_contract = "mimic_stand_target_pd_v2",
            ground_contact_sampling = MjMimicContactContract.Sampling,
            support_bodies = new[] { new[] { "Foot_L", "Toe_L" }, new[] { "Foot_R", "Toe_R" } }
        }));
        Assert.True(MjMimicContactContract.IsCorrected(json.RootElement));
    }

    [Theory]
    [InlineData("mimic_stand_target_pd_v2")]
    [InlineData("unknown")]
    public void MissingOrUnknownSemanticsFailClosed(string schema)
    {
        using var json = JsonDocument.Parse(JsonSerializer.Serialize(new { observation_contract = schema }));
        Assert.Throws<InvalidOperationException>(() => MjMimicContactContract.IsCorrected(json.RootElement));
    }

    [Fact]
    public void LegacyCannotBeSilentlyRelabeled()
    {
        using var json = JsonDocument.Parse("""{"observation_contract":"mimic_stand_target_pd_v1","support_bodies":[]}""");
        Assert.Throws<InvalidOperationException>(() => MjMimicContactContract.IsCorrected(json.RootElement));
    }
}
