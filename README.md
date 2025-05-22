
# Capstone Prototyping Projects

Utilizing Windsurf, AUGMENT, Co Pilot, chat.openai.com and grok.com to code these projects or part of them.

- Declan Heenan

# Projects

### Smaller Prototype Examples

1. **Web Calculator**: This was a simple Python flask application to implement a web calculator.
2. **Web Database Panel**: This was a simple Python flask application to implement a web-based sqlite database editor, similar to a adminstrator database editor panel.
3. **Web Safetensor Model Viewer**: This was a simple Flask web-based graph visualizer for safetensor nn.Modules. This viewed each layer of AI models.

### Medium Prototype Examples

1. **Custom MCP Tools**: Custom MCP tools that AI models (agentic ones) can utilize when doing queries. Implemented using a Python library and includes the following tools:
	CONTEXT:
		i. Create and manage 'memories'
		ii. Create and manage 'context'
		iii. Create and manage 'tasks'
		iv. Create and manage 'milestones'
	DEEP THINK:
		i. 'deep think' about implementing a algorithm
		ii. 'deep think' about code generation
		iii. 'fast think' about code generation
		iv. 'deep think' about code analysis and improvement
		v. 'deep think' about code security analysis
		vi. 'deep think' about system design
		vii. 'deep think' about goal-to-start implementation
	DOCUMENTS:
		i. Query the document database and search for related text to the query
		ii. Index a directory of documents and make them searchable
	MISC:
		i. Get the current time
	SPECIAL:
		i. Transcribe audio
		ii. Transcribe video
	THINK:
		i. 'think' problem solving
		ii. 'think' problem decomposition
		iii. 'think' tool selection
		iv. 'think' solution reflection
		v. 'think' api design
		vi. 'think' trade off thinking
	WEB SEARCH:
		i. download webpage (given url)
		ii. search google
		iii. search wikipedia
		iv. search research papers (various sources)
		v. search github repositories

### Large Prototype Examples

- **Next Frame Predict**: Attempt to re-implement the next-frame-prediction video diffusion model architecture from a research paper - could not test thoroughly but forward pass succeeded.
- **Lightning Image Diffusion**: Attempt to implement a Image Diffusion model of high technicality - could not test thoroughly but forward pass succeeded.

# Notes

I found that, especially with technical projects, creating context for the project is essential to get good outputs.

For example, for the lightning diffusion model, I created a notepad file prompt provided below with all the features and components i wanted,
then i fed that into the copilot and asked it to improve it and format it, repeated that a few times, then incrementally implemented it.

Example prompt when doing the "training script" for lightning diffusion:
```
Use "DPM-Solver++" schedular for fast diffusion.
Use mixed precision training.
Use gradient accumulation.
Use gradient checkpointing.
Use EMA weights.
Use efficient data loading.
Use Learning Rate Scheduling and Warmup.
Use  Early Stopping and Checkpointing.
Use Smaller Model/Parameter Sharing.
Use Profiling and add profilers in the code with options to toggle so the training can be profiled.
Use "Adafactor transformers.optimization.Adafactor".
Use "Patch-based MSELoss on the VAE latent and original/output image with high precision" to get as close to the encoded image as possible.

Use Data Augmentation Techniques & Ideas:
- Geometric Augmentations: Random crop, resize, horizontal/vertical flip, rotation, affine transforms.
- Color Augmentations: Color jitter (brightness, contrast, saturation, hue), grayscale, random gamma.
- Random Erasing: Randomly mask out small patches of the image.
- Gaussian/Poisson Noise: Add random noise to simulate sensor or compression artifacts.
- GridMask: Mask out regions in a grid pattern for regularization.

Make sure to implement all the above, do it step-by-step focusing on a basic training script first, then applying all features on top.
```
