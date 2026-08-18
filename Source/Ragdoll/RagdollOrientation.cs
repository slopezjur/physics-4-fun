namespace Physics4Fun.Ragdoll;

/// <summary>
/// Anatomical body orientation relative to gravity/ground.
/// </summary>
public enum RagdollOrientation
{
    Upright,
    Supine,  // Lying on back (chest facing sky)
    Prone,   // Lying on stomach (chest facing ground)
    Side     // Lying on left/right side
}
