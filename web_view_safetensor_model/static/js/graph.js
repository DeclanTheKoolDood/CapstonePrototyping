// Global variables
let svg, simulation, link, node, zoom;
let graphData = null;
let selectedNode = null;
let isLayeredLayout = false; // Track if we're using layered layout
let layerGroups = {}; // Store nodes by layer
let expandedGroups = new Set(); // Track which layer groups are expanded
let currentFilename = ''; // Current file being visualized
let isLargeModel = false; // Flag for large model optimization

// Initialize the graph visualization
function initGraph(filename) {
    // Reset state
    expandedGroups.clear();
    currentFilename = filename;

    // Show loading state
    document.getElementById('graph-container').innerHTML = `
        <div class="d-flex justify-content-center align-items-center" style="height: 100%;">
            <div class="spinner-border text-light" role="status" style="width: 3rem; height: 3rem;">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="ms-3 text-light">Loading model data...</div>
        </div>
    `;

    // Fetch model data from the API
    fetch(`/api/model/${filename}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                if (data.graph && data.graph.nodes && data.graph.nodes.length > 0) {
                    console.log("Graph data loaded successfully");

                    // Store model metadata
                    graphData = data.graph;
                    isLargeModel = data.graph.is_large_model || false;

                    // Show model summary
                    const totalParams = data.graph.total_params || 0;
                    const numTensors = data.graph.num_tensors || 0;
                    const totalLayers = data.graph.total_layer_groups || 0;
                    const shownLayers = data.graph.shown_layer_groups || 0;
                    const limitedView = data.graph.limited_view || false;

                    document.getElementById('model-summary').innerHTML = `
                        <div class="alert alert-info">
                            <h5 class="mb-2">Model Summary</h5>
                            <div><strong>Parameters:</strong> ${totalParams.toLocaleString()}</div>
                            <div><strong>Tensors:</strong> ${numTensors.toLocaleString()}</div>
                            <div><strong>Layer Groups:</strong> ${limitedView ?
                                `${shownLayers.toLocaleString()} shown (of ${totalLayers.toLocaleString()} total)` :
                                totalLayers.toLocaleString()}</div>
                            ${isLargeModel ? '<div class="mt-2"><i class="fas fa-info-circle"></i> Large model detected. Using optimized rendering.</div>' : ''}
                            ${limitedView ? '<div class="mt-2"><i class="fas fa-exclamation-triangle"></i> Very large model detected. Showing only the most significant layers.</div>' : ''}
                            <div class="mt-2"><i class="fas fa-project-diagram"></i> Showing connections only between first parameters (highlighted with gold borders). Hover over any node to see its specific connections.</div>
                        </div>
                    `;

                    // Render the graph
                    renderGraph(graphData);

                    // For large models, automatically use layered layout
                    if (isLargeModel) {
                        organizeByLayers();
                    }
                } else {
                    showError('No nodes found in the model. The file may not be a valid model or may be empty.');
                }
            } else {
                showError(data.error || 'Failed to load model data');
            }
        })
        .catch(error => {
            console.error("Error loading model data:", error);
            showError('Error loading model data: ' + error);
        });
}

// Render the graph visualization
function renderGraph(data) {
    // Clear any existing graph
    d3.select('#graph-container').html('');

    // Set up the SVG container
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Create SVG element
    svg = d3.select('#graph-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', [0, 0, width, height])
        .attr('style', 'max-width: 100%; height: auto;');

    // Add arrowhead marker definition
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 15)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10,0 L 0,5')
        .attr('fill', '#555')
        .style('stroke', 'none');

    // Add zoom behavior
    zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Create a group for the graph elements
    const g = svg.append('g');

    // Create a tooltip
    const tooltip = d3.select('body')
        .append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0);

    // Create the force simulation
    simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(30));

    // Create the links
    link = g.append('g')
        .attr('class', 'links')
        .selectAll('path')
        .data(data.links)
        .enter()
        .append('path')
        .attr('class', l => `link ${isIndex0Connection(l) ? 'visible' : ''}`)
        .attr('marker-end', 'url(#arrowhead)');

    // Create the nodes
    node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('.node')
        .data(data.nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    // Add different shapes based on node type
    node.each(function(d) {
        const nodeSelection = d3.select(this);

        if (d.is_group) {
            // Layer group node - use a larger hexagon
            const hexagonRadius = 15;
            const hexagonPoints = d3.range(6).map(i => {
                const angle = i * Math.PI / 3;
                return [hexagonRadius * Math.sin(angle), hexagonRadius * Math.cos(angle)];
            });

            nodeSelection.append('polygon')
                .attr('points', hexagonPoints.map(p => p.join(',')).join(' '))
                .style('fill', d => getLayerTypeColor(d.layer_type))
                .style('stroke', '#fff')
                .style('stroke-width', '2px')
                .style('opacity', 0.9)
                .classed('group-node', true);

            // Add a + or - symbol based on expanded state
            nodeSelection.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', '.3em')
                .attr('fill', '#fff')
                .attr('font-size', '12px')
                .attr('font-weight', 'bold')
                .text(d => expandedGroups.has(d.id) ? '−' : '+')
                .classed('group-icon', true);
        } else {
            // Regular tensor node - use a circle
            nodeSelection.append('circle')
                .attr('r', 6)
                .style('fill', d => getLayerTypeColor(d.layer_type))
                .style('opacity', 0.9);
        }
    })
    .on('mouseover', function(event, d) {
            // Show tooltip
            tooltip.transition()
                .duration(200)
                .style('opacity', .9);

            // Format tooltip content
            let content = '';

            if (d.is_group) {
                // Layer group tooltip
                const groupName = d.id.replace('group:', '');
                content = `<strong>Layer Group: ${groupName}</strong><br>`;
                if (d.layer_type && d.layer_type !== 'unknown') {
                    content += `Primary Type: <span style="color: ${getLayerTypeColor(d.layer_type)}">${d.layer_type}</span><br>`;
                }
                content += `Tensors: ${d.num_tensors.toLocaleString()}<br>`;
                content += `Parameters: ${d.num_params.toLocaleString()}<br>`;
                content += `<span class="text-muted">${expandedGroups.has(d.id) ? 'Click to collapse' : 'Click to expand'}</span>`;
            } else {
                // Regular tensor tooltip
                content = `<strong>${d.id}</strong><br>`;
                if (d.layer_type && d.layer_type !== 'unknown') {
                    content += `Layer Type: <span style="color: ${getLayerTypeColor(d.layer_type)}">${d.layer_type}</span><br>`;
                }
                if (d.shape) content += `Shape: ${d.shape}<br>`;
                if (d.dtype) content += `Data Type: ${d.dtype}`;
            }

            tooltip.html(content)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');

            // Highlight connected links
            link.each(function(l) {
                const linkElement = d3.select(this);
                const isDirectConnection = l.source.id === d.id || l.target.id === d.id;

                if (isDirectConnection) {
                    // Direct connections are highlighted
                    linkElement
                        .classed('visible', true)
                        .style('stroke', '#fbbc05')  // Gold color for direct connections
                        .style('stroke-width', '2.5px')
                        .style('opacity', 1.0);
                } else if (isIndex0Connection(l)) {
                    // First parameter connections stay visible but dimmed
                    linkElement
                        .classed('visible', true)
                        .style('stroke', '#4285f4')  // Blue for first parameter connections
                        .style('stroke-width', '1.5px')
                        .style('opacity', 0.3);
                } else {
                    // All other connections are hidden
                    linkElement
                        .classed('visible', false)
                        .style('stroke', '#555')
                        .style('stroke-width', '1.5px')
                        .style('opacity', 0);
                }
            });
        })
        .on('mouseout', function() {
            // Hide tooltip
            tooltip.transition()
                .duration(500)
                .style('opacity', 0);

            // Reset link styles if no node is selected
            if (!selectedNode) {
                link.each(function(l) {
                    const linkElement = d3.select(this);
                    const isFirstParamConnection = isIndex0Connection(l);

                    // Reset to default styles
                    linkElement
                        .classed('visible', isFirstParamConnection)
                        .style('stroke', isFirstParamConnection ? '#4285f4' : '#555')
                        .style('stroke-width', '1.5px')
                        .style('opacity', isFirstParamConnection ? 0.7 : 0);
                });
            }
        })
        .on('click', function(event, d) {
            event.stopPropagation();

            if (d.is_group) {
                // Handle layer group click - expand/collapse
                if (expandedGroups.has(d.id)) {
                    // Collapse this group
                    collapseLayerGroup(d.id);
                } else {
                    // Expand this group
                    expandLayerGroup(d.id);
                }
            } else {
                // Handle regular tensor node click

                // Toggle selection
                if (selectedNode === d) {
                    // Deselect
                    selectedNode = null;
                    d3.selectAll('.node').classed('selected', false);
                    link.style('stroke', '#555')
                        .style('stroke-width', '1.5px')
                        .style('opacity', 0.6);

                    // Clear tensor details
                    document.getElementById('tensor-details').innerHTML = `
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i> Click on a node in the graph to view tensor details.
                        </div>
                    `;
                } else {
                    // Select new node
                    selectedNode = d;
                    d3.selectAll('.node').classed('selected', n => n === d);

                    // Highlight connected links
                    link.each(function(l) {
                        const linkElement = d3.select(this);
                        const isDirectConnection = l.source.id === d.id || l.target.id === d.id;

                        if (isDirectConnection) {
                            // Direct connections are highlighted
                            linkElement
                                .classed('visible', true)
                                .style('stroke', '#fbbc05')  // Gold color for direct connections
                                .style('stroke-width', '2.5px')
                                .style('opacity', 1.0);
                        } else if (isIndex0Connection(l)) {
                            // First parameter connections stay visible but dimmed
                            linkElement
                                .classed('visible', true)
                                .style('stroke', '#4285f4')  // Blue for first parameter connections
                                .style('stroke-width', '1.5px')
                                .style('opacity', 0.3);
                        } else {
                            // All other connections are hidden
                            linkElement
                                .classed('visible', false)
                                .style('stroke', '#555')
                                .style('stroke-width', '1.5px')
                                .style('opacity', 0);
                        }
                    });

                    // Show tensor details
                    showTensorDetails(currentFilename, d);
                }
            }
        });

    // Add labels to nodes
    node.append('text')
        .attr('dy', d => d.is_group ? -20 : -10)
        .attr('text-anchor', 'middle')
        .attr('fill', '#e0e0e0')
        .text(d => {
            if (d.is_group) {
                // For group nodes, show the layer group name
                const groupName = d.id.replace('group:', '');
                return groupName.length > 15 ? groupName.substring(0, 12) + '...' : groupName;
            } else {
                // For regular nodes, show the tensor name
                const name = d.id.split('.').pop();
                return name.length > 15 ? name.substring(0, 12) + '...' : name;
            }
        });

    // Update positions on each tick
    simulation.on('tick', () => {
        link.attr('d', d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const dr = Math.sqrt(dx * dx + dy * dy);

            return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
        });

        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Highlight the first parameter nodes
    highlightFirstParameterNodes();

    // Add click handler to SVG to deselect nodes
    svg.on('click', () => {
        if (selectedNode) {
            selectedNode = null;
            d3.selectAll('.node').classed('selected', false);
            link.style('stroke', '#555')
                .style('stroke-width', '1.5px')
                .style('opacity', 0.6);

            // Clear tensor details
            document.getElementById('tensor-details').innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i> Click on a node in the graph to view tensor details.
                </div>
            `;
        }
    });
}

// Show tensor details in the sidebar
function showTensorDetails(filename, node) {
    const detailsContainer = document.getElementById('tensor-details');

    // Show loading state
    detailsContainer.innerHTML = `
        <div class="d-flex justify-content-center">
            <div class="spinner-border text-light" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;

    // Fetch tensor details from the API
    fetch(`/api/tensor/${filename}/${encodeURIComponent(node.key)}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const stats = data.stats;

                // Format tensor details
                let html = `
                    <h5 class="mb-3">${node.id}</h5>
                    <div class="tensor-property">
                        <div class="property-name">Shape:</div>
                        <div class="property-value">${stats.shape}</div>
                    </div>
                    <div class="tensor-property">
                        <div class="property-name">Data Type:</div>
                        <div class="property-value">${stats.dtype}</div>
                    </div>
                    <div class="tensor-property">
                        <div class="property-name">Elements:</div>
                        <div class="property-value">${stats.size.toLocaleString()}</div>
                    </div>
                `;

                // Add statistics if available
                if (stats.min !== null) {
                    html += `
                        <div class="tensor-property">
                            <div class="property-name">Min Value:</div>
                            <div class="property-value">${stats.min.toExponential(4)}</div>
                        </div>
                        <div class="tensor-property">
                            <div class="property-name">Max Value:</div>
                            <div class="property-value">${stats.max.toExponential(4)}</div>
                        </div>
                        <div class="tensor-property">
                            <div class="property-name">Mean:</div>
                            <div class="property-value">${stats.mean.toExponential(4)}</div>
                        </div>
                        <div class="tensor-property">
                            <div class="property-name">Standard Deviation:</div>
                            <div class="property-value">${stats.std.toExponential(4)}</div>
                        </div>
                    `;
                }

                detailsContainer.innerHTML = html;
            } else {
                showError('Failed to load tensor details: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            showError('Error loading tensor details: ' + error);
        });
}

// Drag functions for nodes
function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);

    // If we're in layered layout, keep the node fixed at its new position
    if (isLayeredLayout) {
        // Keep the node fixed at its current position
        d.fx = d.x;
        d.fy = d.y;
    } else {
        // In free layout, release the node
        d.fx = null;
        d.fy = null;
    }
}

// Zoom functions
function zoomIn() {
    svg.transition().duration(300).call(zoom.scaleBy, 1.5);
}

function zoomOut() {
    svg.transition().duration(300).call(zoom.scaleBy, 0.75);
}

function resetZoom() {
    svg.transition().duration(300).call(zoom.transform, d3.zoomIdentity);
}

// Search nodes in the graph
function searchNodes(searchTerm) {
    if (!graphData) return;

    if (searchTerm === '') {
        // Reset all nodes
        node.style('opacity', 1);
        link.style('opacity', 0.6);
        return;
    }

    // Find matching nodes
    const matchingNodes = graphData.nodes.filter(n =>
        n.id.toLowerCase().includes(searchTerm)
    );

    // Highlight matching nodes and their connections
    node.style('opacity', n => matchingNodes.includes(n) ? 1 : 0.2);

    link.style('opacity', l => {
        const sourceMatches = matchingNodes.some(n => n.id === l.source.id);
        const targetMatches = matchingNodes.some(n => n.id === l.target.id);
        return (sourceMatches || targetMatches) ? 0.8 : 0.1;
    });
}

// Organize nodes by layers
function organizeByLayers() {
    if (!graphData || !graphData.nodes.length) return;

    // Stop the simulation
    simulation.stop();

    // Extract layer information from node IDs
    layerGroups = {};

    graphData.nodes.forEach(node => {
        // Use layer_group if available, otherwise extract from ID
        let layerName;

        if (node.layer_group) {
            // Use the layer_group provided by the backend
            layerName = node.layer_group;
        } else {
            // Extract layer name from the node ID
            const parts = node.id.split('.');
            layerName = parts[0]; // Default to first part

            // Handle special cases like 'layer1', 'bn1', etc.
            if (parts[0].match(/^layer\d+/) || parts[0].match(/^bn\d+/) ||
                parts[0].match(/^conv\d+/) || parts[0].match(/^fc\d+/) ||
                parts[0].match(/^block\d+/)) {
                layerName = parts[0];
            } else if (parts.length > 1) {
                // Try to find a numeric indicator in the parts
                for (let i = 0; i < parts.length; i++) {
                    if (parts[i].match(/\d+/)) {
                        layerName = parts.slice(0, i+1).join('.');
                        break;
                    }
                }
            }
        }

        // Add node to its layer group
        if (!layerGroups[layerName]) {
            layerGroups[layerName] = [];
        }
        layerGroups[layerName].push(node);
    });

    // Sort layers by name
    const sortedLayers = Object.keys(layerGroups).sort();

    // Calculate positions for each layer
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    const layerWidth = width / (sortedLayers.length + 1);

    // Position nodes in each layer
    sortedLayers.forEach((layerName, layerIndex) => {
        const nodesInLayer = layerGroups[layerName];
        const nodeHeight = height / (nodesInLayer.length + 1);

        nodesInLayer.forEach((node, nodeIndex) => {
            // Position node in its layer
            node.fx = (layerIndex + 1) * layerWidth;

            if (node.is_group) {
                // Layer groups are positioned at the top of their column
                node.fy = 80;
            } else if (node.parent_group) {
                // Child nodes of expanded groups are positioned below their parent
                const parentNode = graphData.nodes.find(n => n.id === node.parent_group);
                if (parentNode) {
                    // Calculate position based on parent and number of siblings
                    const siblingNodes = graphData.nodes.filter(n =>
                        !n.is_group && n.parent_group === node.parent_group
                    );
                    const siblingIndex = siblingNodes.indexOf(node);
                    const totalSiblings = siblingNodes.length;

                    // Position in a grid below the parent
                    const rowSize = Math.ceil(Math.sqrt(totalSiblings));
                    const row = Math.floor(siblingIndex / rowSize);
                    const col = siblingIndex % rowSize;

                    node.fx = parentNode.fx + (col - rowSize/2 + 0.5) * 40;
                    node.fy = parentNode.fy + 100 + row * 40;
                } else {
                    // Fallback if parent not found
                    node.fy = (nodeIndex + 1) * nodeHeight;
                }
            } else {
                // Regular nodes are positioned evenly in their column
                node.fy = (nodeIndex + 1) * nodeHeight;
            }
        });
    });

    // Update the visualization
    simulation.alpha(0.3).restart();

    // Set flag to indicate we're using layered layout
    isLayeredLayout = true;

    // Update the button text
    document.getElementById('toggle-layout').textContent = 'Free Layout';

    // Add layer labels if they don't exist
    if (!document.querySelector('.layer-label')) {
        const g = svg.select('g');

        sortedLayers.forEach((layerName, layerIndex) => {
            g.append('text')
                .attr('class', 'layer-label')
                .attr('x', (layerIndex + 1) * layerWidth)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .attr('fill', '#e0e0e0')
                .text(layerName);
        });
    }
}

// Switch to free layout (remove fixed positions)
function switchToFreeLayout() {
    if (!graphData || !graphData.nodes.length) return;

    // Remove fixed positions
    graphData.nodes.forEach(node => {
        node.fx = null;
        node.fy = null;
    });

    // Remove layer labels
    svg.selectAll('.layer-label').remove();

    // Restart simulation with stronger forces
    simulation
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(
            document.getElementById('graph-container').clientWidth / 2,
            document.getElementById('graph-container').clientHeight / 2
        ))
        .alpha(1)
        .restart();

    // Update flag
    isLayeredLayout = false;

    // Update button text
    document.getElementById('toggle-layout').textContent = 'Layer Layout';
}

// Toggle between layered and free layouts
function toggleLayout() {
    if (isLayeredLayout) {
        switchToFreeLayout();
    } else {
        organizeByLayers();
    }
}

// Expand a layer group to show its tensors
function expandLayerGroup(groupId) {
    if (!graphData || !currentFilename) return;

    // Mark this group as expanded
    expandedGroups.add(groupId);

    // Update the group icon
    d3.select(`.node[id="${CSS.escape(groupId)}"] .group-icon`).text('−');

    // Extract the layer group name
    const layerGroup = groupId.replace('group:', '');

    // Show loading indicator
    document.getElementById('tensor-details').innerHTML = `
        <div class="d-flex justify-content-center">
            <div class="spinner-border text-light" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="ms-3">Loading layer details...</div>
        </div>
    `;

    // Fetch the layer group details
    fetch(`/api/layer_group/${currentFilename}/${layerGroup}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                console.log(`Loaded ${data.num_tensors} tensors for layer group ${layerGroup}`);

                // Add the new nodes and links to the graph
                if (data.nodes && data.nodes.length > 0) {
                    // Get the position of the group node
                    const groupNode = graphData.nodes.find(n => n.id === groupId);
                    const groupX = groupNode.x || 0;
                    const groupY = groupNode.y || 0;

                    // Add the new nodes to the graph data
                    const newNodes = data.nodes.map(node => {
                        // Position the new nodes around the group node
                        const angle = Math.random() * 2 * Math.PI;
                        const distance = 50 + Math.random() * 50;
                        return {
                            ...node,
                            x: groupX + distance * Math.cos(angle),
                            y: groupY + distance * Math.sin(angle)
                        };
                    });

                    // Add the new links
                    const newLinks = data.links.map(link => ({
                        ...link,
                        value: 1
                    }));

                    // Add connections from group to its tensors
                    newNodes.forEach(node => {
                        newLinks.push({
                            source: groupId,
                            target: node.id,
                            value: 1
                        });
                    });

                    // Update the graph data
                    graphData.nodes = [...graphData.nodes, ...newNodes];
                    graphData.links = [...graphData.links, ...newLinks];

                    // Re-render the graph
                    renderGraph(graphData);

                    // If we're using layered layout, reapply it
                    if (isLayeredLayout) {
                        organizeByLayers();
                    }

                    // Show success message
                    document.getElementById('tensor-details').innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle me-2"></i> Expanded layer group: ${layerGroup}
                            <div class="mt-2">
                                <strong>Tensors:</strong> ${data.num_tensors.toLocaleString()}<br>
                                <strong>Parameters:</strong> ${data.total_params.toLocaleString()}
                            </div>
                        </div>
                    `;
                } else {
                    // No tensors found
                    document.getElementById('tensor-details').innerHTML = `
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i> No tensors found in layer group: ${layerGroup}
                        </div>
                    `;
                }
            } else {
                showError(data.error || `Failed to load layer group: ${layerGroup}`);
            }
        })
        .catch(error => {
            console.error(`Error loading layer group ${layerGroup}:`, error);
            showError(`Error loading layer group: ${error}`);
        });
}

// Collapse a layer group to hide its tensors
function collapseLayerGroup(groupId) {
    if (!graphData) return;

    // Remove this group from the expanded set
    expandedGroups.delete(groupId);

    // Update the group icon
    d3.select(`.node[id="${CSS.escape(groupId)}"] .group-icon`).text('+');

    // Find all nodes that belong to this group
    const groupTensors = graphData.nodes.filter(n =>
        !n.is_group && n.parent_group === groupId
    );

    // Remove these nodes from the graph data
    graphData.nodes = graphData.nodes.filter(n =>
        n.is_group || n.parent_group !== groupId
    );

    // Remove any links connected to these nodes
    const tensorIds = groupTensors.map(n => n.id);
    graphData.links = graphData.links.filter(l =>
        !tensorIds.includes(l.source.id) && !tensorIds.includes(l.target.id)
    );

    // Re-render the graph
    renderGraph(graphData);

    // If we're using layered layout, reapply it
    if (isLayeredLayout) {
        organizeByLayers();
    }

    // Show message
    const layerGroup = groupId.replace('group:', '');
    document.getElementById('tensor-details').innerHTML = `
        <div class="alert alert-info">
            <i class="fas fa-info-circle me-2"></i> Collapsed layer group: ${layerGroup}
        </div>
    `;
}

// Check if a node is an index 0 node or a group node
function isIndex0OrGroupNode(nodeId) {
    if (!nodeId) return false;

    // Group nodes
    if (nodeId.startsWith('group:')) return true;

    // We want to be very specific about which nodes are considered "index 0"
    // Only consider the first parameter of each layer (typically weight)

    // Common patterns for first parameters in layers
    const firstParamPatterns = [
        /\.0\.weight$/,           // layer.0.weight
        /\.0\.query\.weight$/,    // layer.0.query.weight
        /\.0\.self\.query\.weight$/,  // layer.0.self.query.weight
        /^layer0\.weight$/,       // layer0.weight
        /^conv1\.weight$/,        // conv1.weight
        /^fc1\.weight$/,          // fc1.weight
        /^linear1\.weight$/,      // linear1.weight
        /^bn1\.weight$/,          // bn1.weight
        /^embedding\.weight$/     // embedding.weight
    ];

    // Check if the node ID matches any of the patterns
    return firstParamPatterns.some(pattern => pattern.test(nodeId));
}

// Check if a link connects to index 0 nodes or group nodes
function isIndex0Connection(link) {
    if (!link || !link.source || !link.target) return false;

    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
    const targetId = typeof link.target === 'object' ? link.target.id : link.target;

    // For connections to be visible by default, BOTH nodes must be either:
    // 1. A first parameter node, or
    // 2. A group node

    // At least one must be a first parameter node or group node
    const hasFirstParamOrGroup = isIndex0OrGroupNode(sourceId) || isIndex0OrGroupNode(targetId);

    // If both are group nodes, show the connection
    const bothAreGroups =
        (sourceId && sourceId.startsWith('group:')) &&
        (targetId && targetId.startsWith('group:'));

    // If one is a group and one is its first parameter, show the connection
    const isGroupToFirstParam =
        (sourceId && sourceId.startsWith('group:') && isIndex0OrGroupNode(targetId)) ||
        (targetId && targetId.startsWith('group:') && isIndex0OrGroupNode(sourceId));

    // If both are first parameters in the same layer, show the connection
    // This is for connections like layer.0.weight to layer.0.bias
    const bothAreFirstParams =
        isIndex0OrGroupNode(sourceId) &&
        isIndex0OrGroupNode(targetId) &&
        !sourceId.startsWith('group:') &&
        !targetId.startsWith('group:');

    return bothAreGroups || isGroupToFirstParam || bothAreFirstParams;
}

// Get color for layer type
function getLayerTypeColor(layerType) {
    switch(layerType) {
        case 'conv': return '#4285f4';      // Blue
        case 'linear': return '#0f9d58';    // Green
        case 'batchnorm': return '#f4b400'; // Yellow
        case 'embedding': return '#db4437'; // Red
        case 'recurrent': return '#9c27b0'; // Purple
        case 'attention': return '#ff6d00'; // Orange
        default: return '#4285f4';          // Default blue
    }
}

// Highlight the first parameter nodes to make them more visible
function highlightFirstParameterNodes() {
    if (!node) return;

    // Add a pulsing animation to the first parameter nodes
    node.filter(d => isIndex0OrGroupNode(d.id) && !d.id.startsWith('group:'))
        .select('circle, polygon')
        .classed('first-parameter', true);

    // Log information about which connections are visible
    if (graphData && graphData.links) {
        const totalLinks = graphData.links.length;
        const visibleLinks = graphData.links.filter(l => isIndex0Connection(l)).length;
        console.log(`Showing ${visibleLinks} of ${totalLinks} connections (${Math.round(visibleLinks/totalLinks*100)}%)`);

        // Log some examples of visible connections
        const visibleExamples = graphData.links.filter(l => isIndex0Connection(l)).slice(0, 5);
        console.log("Examples of visible connections:", visibleExamples.map(l => {
            const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
            const targetId = typeof l.target === 'object' ? l.target.id : l.target;
            return `${sourceId} → ${targetId}`;
        }));
    }
}

// Show error message
function showError(message) {
    document.getElementById('tensor-details').innerHTML = `
        <div class="alert alert-danger">
            <i class="fas fa-exclamation-circle me-2"></i> ${message}
        </div>
    `;
}
