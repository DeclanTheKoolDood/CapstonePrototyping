from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from config import Config
import os
import json
import torch
from safetensors.torch import load_file
import networkx as nx

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config['SECRET_KEY']

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def parse_safetensor(file_path):
    """Parse a safetensor file and extract model structure"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': f'File not found: {file_path}'
            }

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return {
                'success': False,
                'error': 'File is empty'
            }

        print(f"Loading safetensor file: {file_path} (size: {file_size / (1024*1024):.2f} MB)")

        # Load the safetensor file
        tensors = load_file(file_path)

        # Check if tensors were loaded
        if not tensors:
            return {
                'success': False,
                'error': 'No tensors found in file'
            }

        print(f"Loaded {len(tensors)} tensors from file")

        # Create a graph representation of the model
        G = nx.DiGraph()

        # Extract tensor information and organize by layers
        tensor_info = {}
        layer_groups = {}
        total_params = 0

        # First pass: identify all layer groups and count parameters
        for key, tensor in tensors.items():
            try:
                # Get tensor shape, dtype, and other metadata
                shape = tensor.shape
                dtype = str(tensor.dtype)
                num_params = tensor.numel()
                total_params += num_params

                # Clean up layer name (remove common prefixes/suffixes)
                parts = key.split('.')

                # Extract layer information for better visualization
                layer_type = "unknown"
                if "weight" in key or "bias" in key:
                    # Try to determine the layer type
                    if any(t in key for t in ["conv", "Conv"]):
                        layer_type = "conv"
                    elif any(t in key for t in ["linear", "Linear", "fc", "FC"]):
                        layer_type = "linear"
                    elif any(t in key for t in ["bn", "BatchNorm"]):
                        layer_type = "batchnorm"
                    elif any(t in key for t in ["embedding", "Embedding"]):
                        layer_type = "embedding"
                    elif any(t in key for t in ["lstm", "LSTM", "gru", "GRU"]):
                        layer_type = "recurrent"
                    elif any(t in key for t in ["attention", "Attention"]):
                        layer_type = "attention"

                # Determine the layer group (for visualization)
                layer_group = parts[0]  # Default to first part

                # Handle common model prefixes
                if len(parts) > 1 and parts[0] in ["model", "module", "encoder", "decoder", "transformer", "backbone"]:
                    layer_group = parts[1]

                # For deeper nested structures, try to find a meaningful group name
                if len(parts) > 2 and parts[1].isdigit() and not parts[0].isdigit():
                    # Handle cases like "layer.0.weight" -> group as "layer"
                    layer_group = parts[0]

                # Store basic tensor info
                tensor_info[key] = {
                    'shape': str(shape),
                    'dtype': dtype,
                    'size': num_params,
                    'layer_group': layer_group,
                    'layer_type': layer_type
                }

                # Add to layer groups
                if layer_group not in layer_groups:
                    layer_groups[layer_group] = {
                        'tensors': [],
                        'total_params': 0,
                        'types': set()
                    }

                layer_groups[layer_group]['tensors'].append(key)
                layer_groups[layer_group]['total_params'] += num_params
                layer_groups[layer_group]['types'].add(layer_type)

            except Exception as tensor_error:
                print(f"Error processing tensor {key}: {str(tensor_error)}")
                # Continue with other tensors even if one fails
                continue

        print(f"Total parameters: {total_params:,}")
        print(f"Identified {len(layer_groups)} layer groups")

        # Second pass: create nodes for each layer group and individual tensors
        # For large models, we'll only create group nodes initially
        is_large_model = len(tensors) > 1000 or total_params > 10000000  # 10M params threshold

        # For extremely large models, limit the number of layer groups shown initially
        max_groups_to_show = 100  # Limit for very large models
        if len(layer_groups) > max_groups_to_show:
            print(f"Very large model detected with {len(layer_groups)} layer groups. Limiting initial view.")
            # Sort layer groups by parameter count (descending) and take the top ones
            sorted_groups = sorted(layer_groups.items(),
                                  key=lambda x: x[1]['total_params'],
                                  reverse=True)[:max_groups_to_show]
            limited_layer_groups = {name: data for name, data in sorted_groups}
        else:
            limited_layer_groups = layer_groups

        # Add layer group nodes
        for group_name, group_data in limited_layer_groups.items():
            # Determine the primary type of this layer group
            primary_type = "unknown"
            if "conv" in group_data['types']:
                primary_type = "conv"
            elif "linear" in group_data['types']:
                primary_type = "linear"
            elif "batchnorm" in group_data['types']:
                primary_type = "batchnorm"
            elif "embedding" in group_data['types']:
                primary_type = "embedding"
            elif "recurrent" in group_data['types']:
                primary_type = "recurrent"
            elif "attention" in group_data['types']:
                primary_type = "attention"

            # Add group node
            G.add_node(
                f"group:{group_name}",
                is_group=True,
                layer_type=primary_type,
                layer_group=group_name,
                num_tensors=len(group_data['tensors']),
                num_params=group_data['total_params'],
                expanded=False  # Initially collapsed
            )

            # For smaller models, also add individual tensor nodes
            if not is_large_model:
                for tensor_key in group_data['tensors']:
                    tensor_info_data = tensor_info[tensor_key]
                    G.add_node(
                        tensor_key,
                        is_group=False,
                        layer_type=tensor_info_data['layer_type'],
                        layer_group=group_name,
                        shape=tensor_info_data['shape'],
                        dtype=tensor_info_data['dtype'],
                        key=tensor_key,
                        parent_group=f"group:{group_name}"
                    )

                    # Connect tensor to its group
                    G.add_edge(f"group:{group_name}", tensor_key)

        # Add connections between layer groups based on model architecture patterns
        group_names = list(limited_layer_groups.keys())
        for i in range(len(group_names) - 1):
            # Connect sequential layer groups
            G.add_edge(f"group:{group_names[i]}", f"group:{group_names[i+1]}")

            # For more complex architectures, we could add additional heuristics here

        # Add a note if we limited the number of layer groups
        if len(layer_groups) > len(limited_layer_groups):
            print(f"Note: Showing {len(limited_layer_groups)} out of {len(layer_groups)} layer groups")

        # Check if we have any nodes
        if len(G.nodes()) == 0:
            return {
                'success': False,
                'error': 'No valid tensors found in file'
            }

        print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")

        # Convert graph to JSON for visualization
        graph_data = {
            'nodes': [{'id': node, 'group': 1, **G.nodes[node]} for node in G.nodes()],
            'links': [{'source': u, 'target': v, 'value': 1} for u, v in G.edges()],
            'is_large_model': is_large_model,
            'total_params': total_params,
            'num_tensors': len(tensors),
            'total_layer_groups': len(layer_groups),
            'shown_layer_groups': len(limited_layer_groups),
            'limited_view': len(layer_groups) > len(limited_layer_groups),
            'layer_groups': {name: {'count': len(data['tensors']), 'params': data['total_params']}
                            for name, data in layer_groups.items()}
        }

        return {
            'graph': graph_data,
            'tensor_info': tensor_info,
            'success': True
        }

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error parsing safetensor file: {str(e)}")
        print(traceback_str)
        return {
            'success': False,
            'error': f"Error: {str(e)}"
        }

@app.route('/')
def index():
    # List available safetensor files in the upload directory
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.safetensors'):
            files.append(filename)

    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        flash(f'File {filename} uploaded successfully!', 'success')
        return redirect(url_for('view_model', filename=filename))

    flash('Invalid file type. Only .safetensors files are allowed.', 'danger')
    return redirect(url_for('index'))

@app.route('/view/<filename>')
def view_model(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

    if not os.path.exists(file_path):
        flash('File not found', 'danger')
        return redirect(url_for('index'))

    return render_template('view_model.html', filename=filename)

@app.route('/api/model/<filename>')
def get_model_data(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})

    result = parse_safetensor(file_path)
    return jsonify(result)

@app.route('/api/tensor/<filename>/<path:tensor_key>')
def get_tensor_details(filename, tensor_key):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})

    try:
        tensors = load_file(file_path)
        if tensor_key in tensors:
            tensor = tensors[tensor_key]

            # Get basic statistics
            stats = {
                'shape': str(tensor.shape),
                'dtype': str(tensor.dtype),
                'size': tensor.numel(),
                'min': float(tensor.min().item()) if tensor.numel() > 0 else None,
                'max': float(tensor.max().item()) if tensor.numel() > 0 else None,
                'mean': float(tensor.mean().item()) if tensor.numel() > 0 else None,
                'std': float(tensor.std().item()) if tensor.numel() > 0 else None
            }

            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Tensor key not found'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/layer_group/<filename>/<layer_group>')
def get_layer_group_details(filename, layer_group):
    """Get details for all tensors in a specific layer group"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})

    try:
        # Load the safetensor file
        tensors = load_file(file_path)

        # Find all tensors that belong to this layer group
        group_tensors = {}
        total_params = 0

        for key, tensor in tensors.items():
            # Extract layer group from tensor key
            parts = key.split('.')
            current_group = parts[0]

            # Handle common model prefixes
            if len(parts) > 1 and parts[0] in ["model", "module", "encoder", "decoder", "transformer", "backbone"]:
                current_group = parts[1]

            # For deeper nested structures
            if len(parts) > 2 and parts[1].isdigit() and not parts[0].isdigit():
                current_group = parts[0]

            # If this tensor belongs to the requested layer group
            if current_group == layer_group:
                # Get tensor metadata
                shape = tensor.shape
                dtype = str(tensor.dtype)
                num_params = tensor.numel()
                total_params += num_params

                # Extract layer type
                layer_type = "unknown"
                if "weight" in key or "bias" in key:
                    if any(t in key for t in ["conv", "Conv"]):
                        layer_type = "conv"
                    elif any(t in key for t in ["linear", "Linear", "fc", "FC"]):
                        layer_type = "linear"
                    elif any(t in key for t in ["bn", "BatchNorm"]):
                        layer_type = "batchnorm"
                    elif any(t in key for t in ["embedding", "Embedding"]):
                        layer_type = "embedding"
                    elif any(t in key for t in ["lstm", "LSTM", "gru", "GRU"]):
                        layer_type = "recurrent"
                    elif any(t in key for t in ["attention", "Attention"]):
                        layer_type = "attention"

                # Store tensor info
                group_tensors[key] = {
                    'shape': str(shape),
                    'dtype': dtype,
                    'size': num_params,
                    'layer_type': layer_type
                }

        if not group_tensors:
            return jsonify({'success': False, 'error': f'No tensors found for layer group: {layer_group}'})

        # Create nodes and links for this layer group
        nodes = []
        links = []

        # Create nodes for each tensor
        for key, info in group_tensors.items():
            nodes.append({
                'id': key,
                'is_group': False,
                'layer_type': info['layer_type'],
                'layer_group': layer_group,
                'shape': info['shape'],
                'dtype': info['dtype'],
                'key': key,
                'parent_group': f"group:{layer_group}"
            })

        # Try to infer connections between tensors in this group
        tensor_keys = list(group_tensors.keys())
        for i, key in enumerate(tensor_keys):
            parts = key.split('.')

            # Connect related tensors (e.g., weight and bias)
            for j in range(i+1, len(tensor_keys)):
                other_key = tensor_keys[j]
                if any(p in other_key for p in parts):
                    links.append({
                        'source': key,
                        'target': other_key,
                        'value': 1
                    })

        return jsonify({
            'success': True,
            'layer_group': layer_group,
            'total_params': total_params,
            'num_tensors': len(group_tensors),
            'nodes': nodes,
            'links': links
        })

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error getting layer group details: {str(e)}")
        print(traceback_str)
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
