"""Generate C# struct offsets for the MuJoCo bridge — derived, never hand-computed.

Reading `mjModel` at guessed offsets returned `nq=52, nv=0, nu=51` against the true `52/51/36`:
plausible, and wrong. Hand arithmetic over a C struct is exactly the kind of silent error this project
keeps getting caught by, so every offset is located empirically instead — each pointer field is found
by scanning the struct for the address of the array the Python bindings already expose.

Regenerate whenever MuJoCo is upgraded; `MjBridge` re-checks the result at load.
"""
import ctypes
import pathlib

import mujoco

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT.parent / "Source" / "RL" / "MuJoCo" / "MjLayout.cs"

# mjData carries a large arena; mjModel is a few kB of scalars and pointers, and scanning past it
# faults, so the two get different spans.
DATA_SPAN = 200_000
MODEL_SPAN = 8_192

DATA_PTRS = ("xpos", "xquat", "xipos", "ctrl", "qpos", "qvel", "xfrc_applied", "cvel", "subtree_com")
# jnt_qposadr / jnt_dofadr locate a joint's slice of qpos and qvel. Assuming the ball
# is "the last joint" would work today and break the moment the rig gains a body.
MODEL_PTRS = ("body_mass", "jnt_qposadr", "jnt_dofadr", "body_rootid", "geom_bodyid")
MODEL_SCALARS = ("nq", "nv", "nu", "nbody")


def find_ptr(struct_addr, span, target):
    """Offset within the struct at which the 8-byte value `target` is stored."""
    raw = ctypes.string_at(struct_addr, span)
    words = (ctypes.c_uint64 * (span // 8)).from_buffer_copy(raw)
    for i, word in enumerate(words):
        if word == target:
            return i * 8
    return None


def find_scalar(struct_addr, span, value):
    raw = ctypes.string_at(struct_addr, span)
    words = (ctypes.c_uint64 * (span // 8)).from_buffer_copy(raw)
    return [i * 8 for i, word in enumerate(words) if word == value]


def pascal(name):
    return "".join(part.capitalize() for part in name.split("_"))


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "dummy.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    scalars, pointers = {}, {}
    for name in MODEL_SCALARS:
        hits = find_scalar(model._address, MODEL_SPAN, int(getattr(model, name)))
        assert hits, "could not locate mjModel." + name
        scalars[name] = hits[0]

    for name in MODEL_PTRS:
        off = find_ptr(model._address, MODEL_SPAN, getattr(model, name).ctypes.data)
        assert off is not None, "could not locate mjModel." + name
        pointers["Model" + pascal(name)] = off

    for name in DATA_PTRS:
        off = find_ptr(data._address, DATA_SPAN, getattr(data, name).ctypes.data)
        assert off is not None, "could not locate mjData." + name
        pointers["Data" + pascal(name)] = off

    # mjContact.dist is its first member (verified against the installed header).
    # NumPy exposes the actual native stride and field addresses, including padding.
    assert data.ncon > 1
    contact = data.contact
    pointers['DataContact'] = find_ptr(data._address, DATA_SPAN, contact.dist.ctypes.data)
    assert pointers['DataContact'] is not None
    contact_stride = contact.dist.strides[0]
    contact_geom = contact.geom.ctypes.data - contact.dist.ctypes.data
    # ncon is int32, not mjtSize. Probe a unique marker without invoking the engine
    # while the local data's count is changed, and restore it even on failure.
    original_count = data.ncon
    try:
        data.ncon = 123456789
        raw = ctypes.string_at(data._address, DATA_SPAN)
        hits = [i for i in range(0, DATA_SPAN - 4, 4)
                if int.from_bytes(raw[i:i + 4], 'little') == data.ncon]
        assert len(hits) == 1, 'ambiguous mjData.ncon offset'
        ncon_offset = hits[0]
    finally:
        data.ncon = original_count

    lines = []
    for name, off in scalars.items():
        lines.append("    /// <summary>Offset of <c>mjModel." + name
                     + "</c> (mjtSize, 8 bytes).</summary>")
        lines.append("    public const int Model" + name[0].upper() + name[1:]
                     + " = " + str(off) + ";")
    lines.append("")
    for name, off in pointers.items():
        struct = "mjModel" if name.startswith("Model") else "mjData"
        field = name[len("Model"):] if name.startswith("Model") else name[len("Data"):]
        lines.append("    /// <summary>Offset of <c>" + struct + "." + field
                     + "</c> (pointer).</summary>")
        lines.append("    public const int " + name + " = " + str(off) + ";")
    lines.extend(['    // Contact count is int32; contact.geom contains two int32 IDs.',
                  f'    public const int DataNcon = {ncon_offset};',
                  f'    public const int ContactStride = {contact_stride};',
                  f'    public const int ContactGeom = {contact_geom};'])

    body = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "namespace Physics4Fun.RL.MuJoCo;\n"
        "\n"
        "/// <summary>\n"
        "/// Byte offsets into MuJoCo's <c>mjModel</c> and <c>mjData</c>, for the P/Invoke bridge.\n"
        "/// </summary>\n"
        "/// <remarks>\n"
        "/// <para><b>Generated by <c>mujoco_rig/gen_offsets.py</c> - do not hand-edit.</b> The offsets\n"
        "/// are located empirically, by scanning each struct for the address of an array the Python\n"
        "/// bindings already expose, because hand-computing C alignment silently produces plausible\n"
        "/// wrong numbers: reading <c>mjModel</c> at 0/4/8 returned nq=52, nv=0, nu=51 against the true\n"
        "/// 52/51/36.</para>\n"
        "/// <para>Generated against MuJoCo " + mujoco.__version__ + ". Regenerate after any upgrade;\n"
        "/// <c>MjBridge</c> verifies them at load and refuses to run if they no longer agree.</para>\n"
        "/// </remarks>\n"
        "internal static class MjLayout\n"
        "{\n"
        "    /// <summary>MuJoCo version these offsets were generated against.</summary>\n"
        "    public const string Version = \"" + mujoco.__version__ + "\";\n"
        "\n"
        + body + "\n"
        "}\n",
        encoding="utf-8")
    print("wrote " + str(OUT))
    print("  mjModel scalars:", scalars)
    print("  pointers:", pointers)


if __name__ == "__main__":
    main()
