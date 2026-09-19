using System;
using System.Globalization;
using System.Xml;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Reads the generated MJCF once, keeping simulation configuration out of rendering.</summary>
internal sealed class MjModelDefinition
{
    internal XmlDocument Document { get; } = new();
    internal double Timestep { get; }
    internal bool HasTorqueActuators { get; }

    internal MjModelDefinition(string path)
    {
        Document.Load(path);
        string? value = Document.SelectSingleNode("/mujoco/option/@timestep")?.Value;
        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double timestep)
            || !double.IsFinite(timestep) || timestep <= 0)
            throw new InvalidOperationException("Generated MJCF must declare a positive timestep.");
        Timestep = timestep;
        HasTorqueActuators = Document.SelectSingleNode("/mujoco/actuator/motor") != null;
    }
}
