using Physics4Fun.Ragdoll.Interfaces;
using Physics4Fun.Ragdoll.Support;
using Xunit;

namespace Physics4Fun.Tests;

public sealed class PlantedLimbLoadDistributionTests
{
    [Fact]
    public void SharesLoadAlongPlantedChainsAndClearsThePreviousTick()
    {
        var pelvis = new FakeBone("Pelvis") { Mass = 10.0f };
        var thigh = new FakeBone("Thigh_L") { Mass = 5.0f, SupportParent = pelvis };
        var foot = new FakeBone("Foot_L")
        {
            Mass = 2.0f, SupportParent = thigh, InContact = true
        };
        ISupportLoadBone[] bones = [pelvis, thigh, foot];
        var distribution = new PlantedLimbLoadDistribution();

        distribution.Distribute(bones);
        Assert.Equal(17.0f, foot.SupportedMassShare);
        Assert.Equal(17.0f, thigh.SupportedMassShare);
        Assert.Equal(0.0f, pelvis.SupportedMassShare);

        foot.InContact = false;
        distribution.Distribute(bones);
        Assert.Equal(0.0f, foot.SupportedMassShare);
        Assert.Equal(0.0f, thigh.SupportedMassShare);
    }

    [Fact]
    public void IgnoresInvalidEndEffectors()
    {
        var foot = new FakeBone("Foot_R") { InContact = true, IsValid = false };
        var distribution = new PlantedLimbLoadDistribution();

        distribution.Distribute(new ISupportLoadBone[] { foot });

        Assert.Equal(0.0f, foot.SupportedMassShare);
    }
}
