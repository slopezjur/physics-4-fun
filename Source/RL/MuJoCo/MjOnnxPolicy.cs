using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Physics4Fun.RL.MuJoCo;

internal interface IMjPolicyInference : IDisposable
{
    void Predict(float[] observation, float[] action);
}

/// <summary>Owns the ONNX session and validates its tensor interface against deployment metadata.</summary>
internal sealed class MjOnnxPolicy : IMjPolicyInference
{
    private readonly InferenceSession _session;
    private readonly string _input;
    private readonly string[] _outputs;

    internal MjOnnxPolicy(byte[] model, MjPolicyContract contract)
    {
        _session = new InferenceSession(model);
        try
        {
            if (_session.InputMetadata.Count != 1 || _session.OutputMetadata.Count != 1)
                throw new InvalidOperationException("Policy must have one input and one output.");
            var input = _session.InputMetadata.Single();
            var output = _session.OutputMetadata.Single();
            Validate(input.Value, contract.NumObs);
            Validate(output.Value, contract.NumActions);
            _input = input.Key;
            _outputs = new[] { output.Key };
        }
        catch { _session.Dispose(); throw; }
    }

    public void Predict(float[] observation, float[] action)
    {
        var tensor = new DenseTensor<float>(observation, new[] { 1, observation.Length });
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_input, tensor) }, _outputs);
        var output = results.Single().AsTensor<float>();
        if (output.Length != action.Length) throw new InvalidOperationException("Policy output width changed.");
        int i = 0;
        foreach (float value in output) action[i++] = value;
    }

    private static void Validate(NodeMetadata metadata, int width)
    {
        if (metadata.ElementType != typeof(float) || metadata.Dimensions.Length != 2
            || (metadata.Dimensions[0] != 1 && metadata.Dimensions[0] != -1)
            || metadata.Dimensions[1] != width)
            throw new InvalidOperationException("Policy tensor shape does not match its contract.");
    }

    public void Dispose() => _session.Dispose();
}
