using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Physics4Fun.RL.MuJoCo;

/// <summary>
/// Raw P/Invoke surface for MuJoCo's C API.
/// </summary>
/// <remarks>
/// <para><b>No GDExtension is required.</b> This project is Godot Mono on net8.0 and MuJoCo ships a C
/// API in <c>mujoco.dll</c>, so the humanoid's physics is reachable from the existing C# assembly with
/// no C++ toolchain. Measured 2026-09-08: 29.1 us per step from C# against 28.2 us from Python, i.e.
/// P/Invoke costs nothing here.</para>
/// <para>The library is resolved through <see cref="SetLibraryDirectory"/> because it does not sit
/// beside the game binary - it ships inside the Python environment.</para>
/// </remarks>
internal static class MjInterop
{
    internal const string Lib = "mujoco";

    /// <summary>mjOBJ_BODY.</summary>
    internal const int ObjBody = 1;

    /// <summary>mjOBJ_JOINT, verified against <c>mujoco.mjtObj</c>.</summary>
    internal const int ObjJoint = 3;
    internal const int ObjGeom = 5;

    /// <summary>
    /// mjOBJ_ACTUATOR. Verified against <c>mujoco.mjtObj</c>, not guessed: 8 is mjOBJ_LIGHT, and
    /// using it made every actuator lookup return -1 so the body stood still while the gait appeared
    /// to run.
    /// </summary>
    internal const int ObjActuator = 19;

    /// <summary><c>mjOBJ_KEY</c> - a keyframe.</summary>
    internal const int ObjKey = 24;

    private static string? _libraryDirectory;
    private static bool _resolverInstalled;
    private static readonly object LibraryLock = new();

    /// <summary>Points the native loader at the directory holding <c>mujoco.dll</c>.</summary>
    internal static void SetLibraryDirectory(string directory)
    {
        lock (LibraryLock)
        {
            directory = ResolveLibraryDirectory(directory);
            if (_resolverInstalled)
            {
                if (!string.Equals(_libraryDirectory, directory, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException("All MuJoCo instances must use the same native library.");
                return;
            }
            _libraryDirectory = directory;

            NativeLibrary.SetDllImportResolver(typeof(MjInterop).Assembly, (name, assembly, path) =>
            {
                if (name != Lib || string.IsNullOrEmpty(_libraryDirectory))
                {
                    return IntPtr.Zero;
                }

                string candidate = Path.Combine(_libraryDirectory, "mujoco.dll");
                return File.Exists(candidate) ? NativeLibrary.Load(candidate) : IntPtr.Zero;
            });
            _resolverInstalled = true;
        }
    }

    /// <summary>Explicit scene setting, environment override, active conda env, then registered env.</summary>
    private static string ResolveLibraryDirectory(string directory)
    {
        if (!string.IsNullOrWhiteSpace(directory)) return Path.GetFullPath(directory);
        string? configured = Environment.GetEnvironmentVariable("P4F_MUJOCO_LIBRARY");
        if (!string.IsNullOrWhiteSpace(configured)) return Path.GetFullPath(configured);
        string? prefix = Environment.GetEnvironmentVariable("CONDA_PREFIX");
        if (!string.IsNullOrWhiteSpace(prefix))
        {
            string candidate = Path.Combine(prefix, "Lib", "site-packages", "mujoco");
            if (File.Exists(Path.Combine(candidate, "mujoco.dll"))) return candidate;
        }
        string registry = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".conda", "environments.txt");
        if (File.Exists(registry))
            foreach (string environment in File.ReadLines(registry))
            {
                if (Path.GetFileName(environment.Trim()) != "env_isaaclab3") continue;
                string candidate = Path.Combine(environment.Trim(), "Lib", "site-packages", "mujoco");
                if (File.Exists(Path.Combine(candidate, "mujoco.dll"))) return candidate;
            }
        return string.Empty; // Let the platform loader resolve a library installed beside the game.
    }

    [DllImport(Lib, CharSet = CharSet.Ansi)]
    internal static extern IntPtr mj_loadXML(string filename, IntPtr vfs, byte[] error, int errorSz);

    [DllImport(Lib)]
    internal static extern IntPtr mj_versionString();

    [DllImport(Lib)]
    internal static extern IntPtr mj_makeData(IntPtr m);

    [DllImport(Lib)]
    internal static extern void mj_step(IntPtr m, IntPtr d);

    [DllImport(Lib)]
    internal static extern void mj_forward(IntPtr m, IntPtr d);

    [DllImport(Lib)]
    internal static extern void mj_contactForce(IntPtr m, IntPtr d, int id, [Out] double[] result);

    [DllImport(Lib)]
    internal static extern void mj_resetData(IntPtr m, IntPtr d);

    [DllImport(Lib)]
    internal static extern void mj_resetDataKeyframe(IntPtr m, IntPtr d, int key);

    [DllImport(Lib)]
    internal static extern void mj_deleteData(IntPtr d);

    [DllImport(Lib)]
    internal static extern void mj_deleteModel(IntPtr m);

    [DllImport(Lib, CharSet = CharSet.Ansi)]
    internal static extern int mj_name2id(IntPtr m, int type, string name);
}
