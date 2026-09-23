using System;
using System.Linq;
using System.Text.Json;

namespace Physics4Fun.RL.MuJoCo;

internal static class MjMimicContactContract
{
    internal const string Sampling = "last_physics_solve; reset_forward; preserve_on_launch";

    internal static bool IsCorrected(JsonElement contract)
    {
        string? schema = contract.GetProperty("observation_contract").GetString();
        if (schema is "mimic_stand_v1" or "mimic_stand_target_pd_v1")
        {
            if (contract.TryGetProperty("ground_contact_sampling", out _) || contract.TryGetProperty("support_bodies", out _))
                throw new InvalidOperationException("Legacy contact semantics cannot be overridden");
            return false;
        }
        if (schema != "mimic_stand_target_pd_v2"
            || !contract.TryGetProperty("ground_contact_sampling", out var sampling) || sampling.GetString() != Sampling
            || !contract.TryGetProperty("support_bodies", out var bodies)
            || bodies.ValueKind != JsonValueKind.Array || bodies.GetArrayLength() != 2
            || bodies.EnumerateArray().Any(row => row.ValueKind != JsonValueKind.Array || row.GetArrayLength() != 2)
            || !bodies.EnumerateArray().SelectMany(row => row.EnumerateArray().Select(value => value.GetString()))
                .SequenceEqual(new[] { "Foot_L", "Toe_L", "Foot_R", "Toe_R" }))
            throw new InvalidOperationException("Unsupported contact observation contract");
        return true;
    }
}
