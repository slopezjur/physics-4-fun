using System;
using System.Numerics;

class Program {
    static void Main() {
        // Create rotation around X
        Matrix4x4 mat = Matrix4x4.CreateRotationX(1.0f);
        Vector3 v = Vector3.Transform(new Vector3(0, -1, 0), mat);
        Console.WriteLine($"Rotated: {v.X}, {v.Y}, {v.Z}");
    }
}
