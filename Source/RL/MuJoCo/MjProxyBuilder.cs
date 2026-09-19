using System;
using System.Collections.Generic;
using System.Globalization;
using System.Xml;
using Godot;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>
/// Builds the Godot meshes that render a MuJoCo body, by reading the model's own MJCF.
/// </summary>
/// <remarks>
/// <para>Separated from <see cref="MujocoDummy"/> because drawing a model and stepping one are
/// different jobs with different reasons to change: the renderer changes when the rig grows a part
/// or a colour, the node changes when the control loop does. Keeping both in one class meant every
/// visual fix - the geoms drawn at their body origin, only the first geom of a body drawn, the
/// capsules laid on their side - was edited in the middle of the physics loop.</para>
/// <para>The shapes are read from the SAME file MuJoCo loads, so a collider and its picture cannot
/// disagree; nothing here is authored twice.</para>
/// </remarks>
internal static class MjProxyBuilder
{
    internal static List<(int Body, Node3D Node)> Build(
        MjBridge bridge, Node3D parent, XmlDocument doc)
    {
        var proxies = new List<(int Body, Node3D Node)>();
        // Godot's own material palette, so the proxy reads as the same character rather than a
        // uniform grey pile - which is most of why the rendered dummy looked wrong.
        var palette = new Dictionary<string, Color>
        {
            ["Head"] = new(0.95f, 0.75f, 0.60f),
            ["Spine"] = new(0.18f, 0.55f, 0.85f),
            ["Chest"] = new(0.18f, 0.55f, 0.85f),
            ["Pelvis"] = new(0.12f, 0.35f, 0.55f),
            ["Foot_L"] = new(0.10f, 0.20f, 0.30f),
            ["Foot_R"] = new(0.10f, 0.20f, 0.30f),
            ["ball"] = new(0.85f, 0.25f, 0.20f),
        };
        var limbColor = new Color(0.20f, 0.68f, 0.78f);

        foreach (XmlNode body in doc.SelectNodes("//body")!)
        {
            string? name = body.Attributes?["name"]?.Value;
            if (name == null)
            {
                continue;
            }

            int id = bridge.BodyId(name);
            if (id < 0)
            {
                continue;
            }

            // **Every geom, not just the first.** A MuJoCo body may carry several - the neck rides
            // on the chest - and `SelectSingleNode` silently rendered only one of them, which is
            // the same class of quiet wrongness as ignoring a geom's offset.
            XmlNodeList? geoms = body.SelectNodes("geom");
            if (geoms == null || geoms.Count == 0)
            {
                continue;
            }

            var pivot = new Node3D { Name = name + "_pivot" };
            parent.AddChild(pivot);
            proxies.Add((id, pivot));

            foreach (XmlNode geom in geoms)
            {

                string type = geom.Attributes?["type"]?.Value ?? "capsule";
                float[] size = Array.ConvertAll(
                    (geom.Attributes?["size"]?.Value ?? "0.05").Split(' ', StringSplitOptions.RemoveEmptyEntries),
                    s => float.Parse(s, CultureInfo.InvariantCulture));

                Mesh mesh = type switch
                {
                    // MuJoCo box size is a half-extent; Godot's BoxMesh takes the full size. The model
                    // is authored in MuJoCo's frame, so the extents are permuted back the same way
                    // positions are: godot = (-my, mz, -mx).
                    "box" => new BoxMesh { Size = new Vector3(size[1] * 2.0f, size[2] * 2.0f, size[0] * 2.0f) },
                    "sphere" => new SphereMesh { Radius = size[0], Height = size[0] * 2.0f },
                    // Godot has no ellipsoid primitive; a unit sphere scaled per axis is the same
                    // surface. The scale is applied below, permuted the same way every extent is.
                    "ellipsoid" => new SphereMesh { Radius = 1.0f, Height = 2.0f },
                    _ => new CapsuleMesh { Radius = size[0], Height = (size[1] * 2.0f) + (size[0] * 2.0f) },
                };

                // **No rotation.** A MuJoCo capsule runs along its local Z, and under the basis change
                // `R_godot = M^T R_mj M` a local vector maps as `M^T v`, so `M^T(0,0,1) = (0,1,0)` -
                // Godot's Y, which is already the axis `CapsuleMesh` uses. Rotating anyway laid every
                // limb on its side, which is what made the rendered dummy look like scattered parts.
                var material = new StandardMaterial3D
                {
                    AlbedoColor = palette.TryGetValue(name, out Color c) ? c : limbColor,
                };
                // **The geom's own offset within its body.** Ignoring this was harmless while every
                // geom sat on its body origin, and became visible the moment the foot was split: the
                // foot geom sits 3 cm back and the toe 3 cm forward, so drawing both at their body
                // origins overlapped them by 6 cm. The physics was always right - only the picture was
                // wrong, which is the most expensive kind of wrong to leave in place.
                Vector3 geomOffset = Vector3.Zero;
                string? offsetAttr = geom.Attributes?["pos"]?.Value;
                if (!string.IsNullOrWhiteSpace(offsetAttr))
                {
                    float[] o = Array.ConvertAll(
                        offsetAttr.Split(' ', StringSplitOptions.RemoveEmptyEntries),
                        s => float.Parse(s, CultureInfo.InvariantCulture));
                    // Same frame map as every other position: godot = (-my, mz, -mx).
                    geomOffset = new Vector3(-o[1], o[2], -o[0]);
                }

                var node = new MeshInstance3D
                {
                    Mesh = mesh,
                    MaterialOverride = material,
                    Name = geom.Attributes?["name"]?.Value ?? name,
                    Position = geomOffset,
                };
                if (type == "ellipsoid")
                {
                    // MuJoCo size is (rx, ry, rz); Godot's axes are the permuted (-my, mz, -mx).
                    node.Scale = new Vector3(size[1], size[2], size[0]);
                }

                pivot.AddChild(node);
                if (name == "Head")
                {
                    AddFace(node);
                }
            }
        }

        return proxies;
    }

    /// <summary>
    /// Gives the proxy head the same two eyes and mouth the authored ragdoll has.
    /// </summary>
    /// <remarks>
    /// <para>Cosmetic, but it earns its place: without a face there is no way to tell at a glance
    /// which way the dummy is FACING, and "the feet look like they point the wrong way" is exactly
    /// the kind of report that needs a visible front to check against. The offsets are copied from
    /// <c>Scenes/ActiveRagdoll.tscn</c> so the MuJoCo proxy and the Jolt ragdoll read identically.</para>
    /// <para>Forward is Godot -Z, which is MuJoCo +X - the axis the foot's long side runs along. A
    /// face pointing the other way would mean the frame map is wrong, so this doubles as a check.</para>
    /// </remarks>
    private static void AddFace(Node3D head)
    {
        var dark = new StandardMaterial3D { AlbedoColor = new Color(0.05f, 0.05f, 0.05f), Roughness = 0.6f };
        var eye = new SphereMesh { Radius = 0.022f, Height = 0.044f };
        var mouth = new BoxMesh { Size = new Vector3(0.09f, 0.015f, 0.02f) };

        foreach ((string name, Vector3 at, Mesh shape) in new[]
        {
            ("Eye_L", new Vector3(0.05f, 0.03f, -0.135f), (Mesh)eye),
            ("Eye_R", new Vector3(-0.05f, 0.03f, -0.135f), (Mesh)eye),
            ("Mouth", new Vector3(0.0f, -0.05f, -0.135f), (Mesh)mouth),
        })
        {
            head.AddChild(new MeshInstance3D
            {
                Name = name,
                Mesh = shape,
                MaterialOverride = dark,
                Position = at,
            });
        }
    }
}
