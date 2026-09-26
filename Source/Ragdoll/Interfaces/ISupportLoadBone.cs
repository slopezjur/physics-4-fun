namespace Physics4Fun.Ragdoll.Interfaces;

/// <summary>Bone hierarchy and load share used by a support-load policy.</summary>
public interface ISupportLoadBone : IBoneState
{
    ISupportLoadBone? SupportParent { get; }
    float SupportedMassShare { get; set; }
}
