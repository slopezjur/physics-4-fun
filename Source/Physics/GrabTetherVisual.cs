using Godot;

namespace Physics4Fun.Physics;

internal sealed class GrabTetherVisual
{
    private readonly ImmediateMesh _mesh = new();

    public GrabTetherVisual(Node owner)
    {
        var material = new StandardMaterial3D
        {
            ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded,
            AlbedoColor = new Color(0.2f, 0.95f, 1.0f, 0.9f),
            Transparency = BaseMaterial3D.TransparencyEnum.Alpha
        };
        owner.AddChild(new MeshInstance3D
        {
            Mesh = _mesh,
            MaterialOverride = material,
            CastShadow = GeometryInstance3D.ShadowCastingSetting.Off
        });
    }

    public void Draw(Vector3 start, Vector3 end)
    {
        _mesh.ClearSurfaces();
        _mesh.SurfaceBegin(Mesh.PrimitiveType.Lines);
        _mesh.SurfaceAddVertex(start);
        _mesh.SurfaceAddVertex(end);
        _mesh.SurfaceEnd();
    }

    public void Clear() => _mesh.ClearSurfaces();
}
