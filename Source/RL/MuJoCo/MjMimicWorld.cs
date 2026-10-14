using System;
using System.IO;
using System.Linq;
using System.Xml;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>Character-only policy state inside an optional projectile world.</summary>
internal sealed class MjMimicWorld
{
    private readonly MjBridge _bridge;
    private readonly double[]? _fullQ, _fullV;
    internal string CharacterPath { get; }
    internal int CharacterBodyCount { get; }

    internal MjMimicWorld(MjBridge bridge, string modelPath)
    {
        _bridge = bridge;
        bool ball = bridge.BodyId("ball") >= 0;
        CharacterPath = ball ? Path.Combine(Path.GetDirectoryName(modelPath)!, "dummy.xml") : modelPath;
        CharacterBodyCount = ball ? bridge.BodyId("ball") : bridge.BodyCount;
        bridge.ComBodyCount = CharacterBodyCount;
        if (!ball) return;
        var world = new XmlDocument(); world.Load(modelPath);
        var character = new XmlDocument(); character.Load(CharacterPath);
        var projectiles = world.SelectNodes("/mujoco/worldbody/body")!.Cast<XmlNode>()
            .Where(n => n.Attributes?["name"]?.Value == "ball"
                || n.Attributes?["name"]?.Value?.StartsWith("ball_", StringComparison.Ordinal) == true).ToArray();
        for (int i = 0; i < projectiles.Length; i++)
        {
            string suffix = i == 0 ? string.Empty : $"_{i}";
            if (projectiles[i].Attributes?["name"]?.Value != "ball" + suffix
                || bridge.BodyId("ball" + suffix) != 19 + i
                || bridge.JointId("ball_free" + suffix) != 40 + i
                || projectiles[i].SelectSingleNode("freejoint/@name")?.Value != "ball_free" + suffix)
                throw new InvalidOperationException("Projectiles must be ordered free bodies after the character.");
            projectiles[i].ParentNode!.RemoveChild(projectiles[i]);
        }
        foreach (var doc in new[] { world, character })
        {
            doc.DocumentElement!.RemoveAttribute("model");
            var key = doc.SelectSingleNode("/mujoco/keyframe");
            key?.ParentNode!.RemoveChild(key);
            foreach (XmlNode comment in doc.SelectNodes("//comment()")!.Cast<XmlNode>().ToArray())
                comment.ParentNode!.RemoveChild(comment);
        }
        if (world.DocumentElement!.OuterXml != character.DocumentElement!.OuterXml
            || CharacterBodyCount != 19 || projectiles.Length != bridge.BodyCount - 19
            || bridge.PositionCount != 46 + projectiles.Length * 7
            || bridge.VelocityCount != 45 + projectiles.Length * 6)
            throw new InvalidOperationException("Ball world must append a projectile without changing the character.");
        _fullQ = new double[bridge.PositionCount]; _fullV = new double[bridge.VelocityCount];
        bridge.ReadNativeState(_fullQ, _fullV); // Validate the complete native layout.
    }

    internal void Read(double[] q, double[] v)
    {
        if (_fullQ == null) { _bridge.ReadNativeState(q, v); return; }
        _bridge.ReadNativeState(_fullQ, _fullV!);
        Array.Copy(_fullQ, q, q.Length); Array.Copy(_fullV!, v, v.Length);
    }

    internal void Reset(double[] q, double[] v)
    {
        if (_fullQ == null) { _bridge.ResetNativeState(q, v); return; }
        Array.Clear(_fullQ); Array.Clear(_fullV!);
        Array.Copy(q, _fullQ, q.Length); Array.Copy(v, _fullV!, v.Length);
        for (int i = 0; i < (_fullQ.Length - 46) / 7; i++)
        {
            int start = 46 + 7 * i;
            _fullQ[start] = 40 + .4 * (i % 8);
            _fullQ[start + 1] = 40 + .4 * (i / 8);
            _fullQ[start + 2] = 2;
            _fullQ[start + 3] = 1;
        }
        _bridge.ResetNativeState(_fullQ, _fullV!);
    }
}
