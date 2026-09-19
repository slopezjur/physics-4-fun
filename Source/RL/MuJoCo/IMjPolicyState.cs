using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>The simulation surface needed by a policy, independent of native handles and scenes.</summary>
internal interface IMjPolicyState
{
    int BodyId(string name);
    int JointId(string name);
    Transform3D BodyTransform(int body);
    Vector3 BodySpatialLinearVelocity(int body);
    Vector3 BodyAngularVelocity(int body);
    double JointPosition(int joint);
    double JointVelocity(int joint);
}

/// <summary>The write capability used by the policy loop, separate from observation consumers.</summary>
internal interface IMjPolicyPlant : IMjPolicyState
{
    int ActuatorId(string name);
    void SetControl(int actuator, double value);
}
