import torch
from safetensors.torch import save_file
import os

def create_sample_model():
    """Create a sample model and save it as a safetensor file"""
    # Create a directory for the sample
    os.makedirs('uploads', exist_ok=True)

    # Create some sample tensors that mimic a small neural network
    model_tensors = {
        # Embedding layer
        'embedding.weight': torch.randn(100, 32),

        # Encoder layers
        'encoder.layer0.attention.self.query.weight': torch.randn(64, 32),
        'encoder.layer0.attention.self.query.bias': torch.randn(64),
        'encoder.layer0.attention.self.key.weight': torch.randn(64, 32),
        'encoder.layer0.attention.self.key.bias': torch.randn(64),
        'encoder.layer0.attention.self.value.weight': torch.randn(64, 32),
        'encoder.layer0.attention.self.value.bias': torch.randn(64),
        'encoder.layer0.attention.output.dense.weight': torch.randn(32, 64),
        'encoder.layer0.attention.output.dense.bias': torch.randn(32),
        'encoder.layer0.attention.output.LayerNorm.weight': torch.randn(32),
        'encoder.layer0.attention.output.LayerNorm.bias': torch.randn(32),
        'encoder.layer0.intermediate.dense.weight': torch.randn(128, 32),
        'encoder.layer0.intermediate.dense.bias': torch.randn(128),
        'encoder.layer0.output.dense.weight': torch.randn(32, 128),
        'encoder.layer0.output.dense.bias': torch.randn(32),
        'encoder.layer0.output.LayerNorm.weight': torch.randn(32),
        'encoder.layer0.output.LayerNorm.bias': torch.randn(32),

        'encoder.layer1.attention.self.query.weight': torch.randn(64, 32),
        'encoder.layer1.attention.self.query.bias': torch.randn(64),
        'encoder.layer1.attention.self.key.weight': torch.randn(64, 32),
        'encoder.layer1.attention.self.key.bias': torch.randn(64),
        'encoder.layer1.attention.self.value.weight': torch.randn(64, 32),
        'encoder.layer1.attention.self.value.bias': torch.randn(64),
        'encoder.layer1.attention.output.dense.weight': torch.randn(32, 64),
        'encoder.layer1.attention.output.dense.bias': torch.randn(32),
        'encoder.layer1.attention.output.LayerNorm.weight': torch.randn(32),
        'encoder.layer1.attention.output.LayerNorm.bias': torch.randn(32),
        'encoder.layer1.intermediate.dense.weight': torch.randn(128, 32),
        'encoder.layer1.intermediate.dense.bias': torch.randn(128),
        'encoder.layer1.output.dense.weight': torch.randn(32, 128),
        'encoder.layer1.output.dense.bias': torch.randn(32),
        'encoder.layer1.output.LayerNorm.weight': torch.randn(32),
        'encoder.layer1.output.LayerNorm.bias': torch.randn(32),

        # Convolutional backbone
        'backbone.conv1.weight': torch.randn(32, 3, 3, 3),
        'backbone.conv1.bias': torch.randn(32),
        'backbone.bn1.weight': torch.randn(32),
        'backbone.bn1.bias': torch.randn(32),
        'backbone.bn1.running_mean': torch.randn(32),
        'backbone.bn1.running_var': torch.randn(32),

        'backbone.conv2.weight': torch.randn(64, 32, 3, 3),
        'backbone.conv2.bias': torch.randn(64),
        'backbone.bn2.weight': torch.randn(64),
        'backbone.bn2.bias': torch.randn(64),
        'backbone.bn2.running_mean': torch.randn(64),
        'backbone.bn2.running_var': torch.randn(64),

        'backbone.conv3.weight': torch.randn(128, 64, 3, 3),
        'backbone.conv3.bias': torch.randn(128),
        'backbone.bn3.weight': torch.randn(128),
        'backbone.bn3.bias': torch.randn(128),
        'backbone.bn3.running_mean': torch.randn(128),
        'backbone.bn3.running_var': torch.randn(128),

        # Decoder layers
        'decoder.layer0.self_attention.query.weight': torch.randn(64, 32),
        'decoder.layer0.self_attention.query.bias': torch.randn(64),
        'decoder.layer0.self_attention.key.weight': torch.randn(64, 32),
        'decoder.layer0.self_attention.key.bias': torch.randn(64),
        'decoder.layer0.self_attention.value.weight': torch.randn(64, 32),
        'decoder.layer0.self_attention.value.bias': torch.randn(64),
        'decoder.layer0.self_attention.output.weight': torch.randn(32, 64),
        'decoder.layer0.self_attention.output.bias': torch.randn(32),
        'decoder.layer0.cross_attention.query.weight': torch.randn(64, 32),
        'decoder.layer0.cross_attention.query.bias': torch.randn(64),
        'decoder.layer0.cross_attention.key.weight': torch.randn(64, 32),
        'decoder.layer0.cross_attention.key.bias': torch.randn(64),
        'decoder.layer0.cross_attention.value.weight': torch.randn(64, 32),
        'decoder.layer0.cross_attention.value.bias': torch.randn(64),
        'decoder.layer0.cross_attention.output.weight': torch.randn(32, 64),
        'decoder.layer0.cross_attention.output.bias': torch.randn(32),
        'decoder.layer0.feed_forward.weight1': torch.randn(128, 32),
        'decoder.layer0.feed_forward.bias1': torch.randn(128),
        'decoder.layer0.feed_forward.weight2': torch.randn(32, 128),
        'decoder.layer0.feed_forward.bias2': torch.randn(32),

        # Classification head
        'classifier.weight': torch.randn(10, 32),
        'classifier.bias': torch.randn(10),
    }

    # Save the tensors to a safetensor file
    save_file(model_tensors, 'uploads/sample_model.safetensors')
    print(f"Created sample model with {len(model_tensors)} tensors")
    print("Saved to uploads/sample_model.safetensors")

    # Create a larger model for testing optimization
    create_large_model()

def create_large_model():
    """Create a larger model with many layers to test optimization"""
    print("Creating large model for testing...")

    # Create a large model with many layers
    large_model = {}

    # Create a model with 12 transformer layers, each with multiple components
    num_layers = 12
    hidden_size = 768
    intermediate_size = 3072
    num_attention_heads = 12

    # Add embedding layers
    large_model['embeddings.word_embeddings.weight'] = torch.randn(30000, hidden_size)
    large_model['embeddings.position_embeddings.weight'] = torch.randn(512, hidden_size)
    large_model['embeddings.token_type_embeddings.weight'] = torch.randn(2, hidden_size)
    large_model['embeddings.LayerNorm.weight'] = torch.randn(hidden_size)
    large_model['embeddings.LayerNorm.bias'] = torch.randn(hidden_size)

    # Add transformer layers
    for i in range(num_layers):
        # Self-attention
        large_model[f'encoder.layer.{i}.attention.self.query.weight'] = torch.randn(hidden_size, hidden_size)
        large_model[f'encoder.layer.{i}.attention.self.query.bias'] = torch.randn(hidden_size)
        large_model[f'encoder.layer.{i}.attention.self.key.weight'] = torch.randn(hidden_size, hidden_size)
        large_model[f'encoder.layer.{i}.attention.self.key.bias'] = torch.randn(hidden_size)
        large_model[f'encoder.layer.{i}.attention.self.value.weight'] = torch.randn(hidden_size, hidden_size)
        large_model[f'encoder.layer.{i}.attention.self.value.bias'] = torch.randn(hidden_size)

        # Attention output
        large_model[f'encoder.layer.{i}.attention.output.dense.weight'] = torch.randn(hidden_size, hidden_size)
        large_model[f'encoder.layer.{i}.attention.output.dense.bias'] = torch.randn(hidden_size)
        large_model[f'encoder.layer.{i}.attention.output.LayerNorm.weight'] = torch.randn(hidden_size)
        large_model[f'encoder.layer.{i}.attention.output.LayerNorm.bias'] = torch.randn(hidden_size)

        # Intermediate (FFN)
        large_model[f'encoder.layer.{i}.intermediate.dense.weight'] = torch.randn(intermediate_size, hidden_size)
        large_model[f'encoder.layer.{i}.intermediate.dense.bias'] = torch.randn(intermediate_size)

        # Output
        large_model[f'encoder.layer.{i}.output.dense.weight'] = torch.randn(hidden_size, intermediate_size)
        large_model[f'encoder.layer.{i}.output.dense.bias'] = torch.randn(hidden_size)
        large_model[f'encoder.layer.{i}.output.LayerNorm.weight'] = torch.randn(hidden_size)
        large_model[f'encoder.layer.{i}.output.LayerNorm.bias'] = torch.randn(hidden_size)

    # Add pooler
    large_model['pooler.dense.weight'] = torch.randn(hidden_size, hidden_size)
    large_model['pooler.dense.bias'] = torch.randn(hidden_size)

    # Add classifier
    large_model['classifier.weight'] = torch.randn(2, hidden_size)
    large_model['classifier.bias'] = torch.randn(2)

    # Save the large model
    save_file(large_model, 'uploads/large_model.safetensors')

    # Calculate total parameters
    total_params = sum(tensor.numel() for tensor in large_model.values())

    print(f"Created large model with {len(large_model)} tensors")
    print(f"Total parameters: {total_params:,}")
    print("Saved to uploads/large_model.safetensors")

if __name__ == "__main__":
    create_sample_model()
