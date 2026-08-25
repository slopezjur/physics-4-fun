using System.Globalization;
using Godot;

namespace Physics4Fun.RL;

/// <summary>
/// Reads the <c>--name=value</c> arguments rl/train.py forwards into each Godot process.
///
/// The transport is implicit and worth stating once, here, rather than rediscovering it at each
/// call site: <c>GodotEnv.__init__</c> takes <c>**kwargs</c> and hands every key it does not
/// recognise to <c>_launch_env</c>, which appends it to the child process command line as
/// <c>--key=value</c>. So a kwarg added in rl/train.py arrives here with no plumbing in between,
/// and equally with no schema, no validation and no error if the name is misspelled on either
/// side - a typo reads as "argument absent" and silently takes the default.
///
/// Parsed with InvariantCulture, NOT the machine's culture, and that is not a detail. Python writes
/// "0.908"; this project is developed on a Spanish-locale machine where the decimal separator is a
/// comma, so a culture-sensitive parse either fails outright or - worse - reads "0.908" as 908. A
/// silently thousand-fold curriculum floor is exactly the class of bug that looks like a physics
/// problem for a day. The integer path uses the same culture for the same reason: digit group
/// separators differ too.
///
/// Both methods return null for "absent or unparseable" and leave the policy for that to the
/// caller, because the two cases want different handling depending on the argument - a missing
/// curriculum floor is normal, a missing dummy count is normal, but an unparseable one is a
/// launcher bug the caller should be loud about.
/// </summary>
internal static class CmdlineArgs
{
    /// <summary>The float at <c>--name=value</c>, or null if absent or unparseable.</summary>
    public static float? ReadFloat(string name)
    {
        string? raw = ReadRaw(name);
        if (raw == null)
        {
            return null;
        }

        return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float value)
            ? value
            : null;
    }

    /// <summary>The int at <c>--name=value</c>, or null if absent or unparseable.</summary>
    public static int? ReadInt(string name)
    {
        string? raw = ReadRaw(name);
        if (raw == null)
        {
            return null;
        }

        return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
            ? value
            : null;
    }

    /// <summary>
    /// The raw text after <c>--name=</c>, or null if no argument carries that prefix.
    ///
    /// First match wins. Duplicates are not diagnosed because they cannot occur through the only
    /// path that produces these: a Python dict cannot hold the same key twice.
    /// </summary>
    private static string? ReadRaw(string name)
    {
        string prefix = $"--{name}=";
        foreach (string arg in OS.GetCmdlineArgs())
        {
            if (arg.StartsWith(prefix, System.StringComparison.Ordinal))
            {
                return arg[prefix.Length..];
            }
        }

        return null;
    }
}
