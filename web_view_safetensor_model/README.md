# SafeTensor Model Viewer

A dark-themed web application for visualizing PyTorch models stored in the SafeTensor format. This tool provides an interactive graph visualization of model layers with hoverable and selectable nodes.

## Features

- 🌙 Dark theme UI with modern aesthetics
- 📊 Interactive graph visualization of model layers
- 🔍 Hover over nodes to see tensor details
- 🖱️ Click on nodes to view in-depth tensor statistics
- 🔎 Search functionality to find specific tensors
- 🔄 Zoom and pan controls for exploring large models
- 📱 Responsive design for all devices

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`
3. Upload a SafeTensor file or select an existing one
4. Explore the model structure in the interactive graph view

## About SafeTensors

SafeTensors is a simple format for storing tensors safely (no arbitrary code execution) and efficiently (mmap). It's designed to be a safer alternative to pickle for storing ML weights. Learn more at [huggingface/safetensors](https://github.com/huggingface/safetensors).

## Technologies Used

- Flask: Web framework
- PyTorch: For tensor operations
- safetensors: For reading SafeTensor files
- D3.js: For interactive graph visualization
- Bootstrap 5: Front-end framework with dark theme
- Font Awesome: Icons

## License

MIT
