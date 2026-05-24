// Navigation & Tabs
function switchTab(tabId, event) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabId).classList.add('active');
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }
}

function switchInputType(type, event) {
    document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.input-area').forEach(a => a.classList.remove('active'));
    
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }
    document.getElementById(`input-${type}-area`).classList.add('active');
}

function switchResultTab(tabId, event) {
    document.querySelectorAll('.res-nav-btn').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.result-tab').forEach(a => a.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }
}

// The "Big 20" Reference Library Data
const referenceLibrary = [
    // --- Computer Vision ---
    { id: "resnet", category: "Vision", title: "ResNet-18", author: "He et al. (2015)", desc: "Deep Residual Learning for Image Recognition. Introduces skip connections.", spec: "ResNet-18 with Conv2D(64), max pooling, 4 residual blocks, and a linear classification head." },
    { id: "unet", category: "Vision", title: "U-Net", author: "Ronneberger et al. (2015)", desc: "Encoder-decoder with spatial skip connections for biomedical segmentation.", spec: "U-Net with encoder path (Conv2D and MaxPool), symmetric decoder path (UpSample), and skip connections between them." },
    { id: "vit", category: "Vision", title: "Vision Transformer", author: "Dosovitskiy et al. (2020)", desc: "An Image is Worth 16x16 Words. Treats image patches as sequence tokens.", spec: "Vision Transformer with PatchEmbedding(16,768) followed by 12 TransformerEncoder blocks and a SequencePooling feature aggregator." },
    { id: "yolo", category: "Vision", title: "YOLOv3", author: "Redmon et al. (2018)", desc: "Real-Time Object Detection using a single neural network evaluation.", spec: "Darknet-53 backbone with Conv2D layers, residual blocks, and multi-scale detection heads." },
    { id: "convnext", category: "Vision", title: "ConvNeXt", author: "Liu et al. (2022)", desc: "A ConvNet for the 2020s. Modernizes CNNs using Transformer design choices.", spec: "ConvNeXt with 4 stages of depthwise Conv2D blocks, LayerNorm, and GELU activations." },
    { id: "densenet", category: "Vision", title: "DenseNet", author: "Huang et al. (2017)", desc: "Densely Connected Convolutional Networks. Connects each layer to every other layer.", spec: "DenseNet with initial Conv2D, 4 Dense Blocks (with dense skip connections), transition layers, and global average pooling." },
    { id: "efficientnet", category: "Vision", title: "EfficientNet", author: "Tan & Le (2019)", desc: "Rethinking Model Scaling for Convolutional Neural Networks.", spec: "EfficientNet backbone with MBConv blocks, Squeeze-and-Excitation, and Swish activation." },
    { id: "mobilenet", category: "Vision", title: "MobileNetV2", author: "Sandler et al. (2018)", desc: "Inverted Residuals and Linear Bottlenecks for mobile vision.", spec: "MobileNetV2 with inverted residual blocks, depthwise separable Conv2D, and linear bottlenecks." },
    
    // --- NLP / Transformers ---
    { id: "transformer", category: "NLP", title: "Transformer", author: "Vaswani et al. (2017)", desc: "Attention Is All You Need. Replaces RNNs with multi-head self-attention.", spec: "Standard Transformer with a 6-layer TransformerEncoder and a 6-layer TransformerDecoder using CrossAttention." },
    { id: "bert", category: "NLP", title: "BERT", author: "Devlin et al. (2018)", desc: "Bidirectional Encoder Representations from Transformers.", spec: "BERT Base with 12 TransformerEncoder blocks, 12 attention heads, and hidden size of 768." },
    { id: "gpt2", category: "NLP", title: "GPT-2", author: "Radford et al. (2019)", desc: "Language Models are Unsupervised Multitask Learners.", spec: "GPT-2 with 12 TransformerDecoder blocks using CausalAttention and masked self-attention." },
    { id: "llama", category: "NLP", title: "Llama 2", author: "Touvron et al. (2023)", desc: "Open Foundation and Fine-Tuned Chat Models.", spec: "Llama 2 architecture with RMSNorm, SwiGLU activation, and Rotary Positional Embeddings (RoPE) in Decoder blocks." },
    { id: "roberta", category: "NLP", title: "RoBERTa", author: "Liu et al. (2019)", desc: "A Robustly Optimized BERT Pretraining Approach.", spec: "RoBERTa with 12 TransformerEncoder blocks using dynamic masking and GELU." },
    { id: "t5", category: "NLP", title: "T5", author: "Raffel et al. (2019)", desc: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.", spec: "T5 Encoder-Decoder architecture with a shared vocabulary, 12 Encoder blocks, and 12 Decoder blocks with CrossAttention." },
    { id: "bart", category: "NLP", title: "BART", author: "Lewis et al. (2019)", desc: "Denoising Sequence-to-Sequence Pre-training.", spec: "BART Base: 6-layer bidirectional TransformerEncoder producing memory states, 6-layer autoregressive TransformerDecoder with CrossAttention to encoder memory." },

    // --- Generative Models ---
    { id: "ddpm", category: "Generative", title: "DDPM", author: "Ho et al. (2020)", desc: "Denoising Diffusion Probabilistic Models.", spec: "Diffusion UNet with time embeddings, spatial downsampling, self-attention at low resolutions, and upsampling." },
    { id: "stablediffusion", category: "Generative", title: "Stable Diffusion", author: "Rombach et al. (2022)", desc: "High-Resolution Image Synthesis with Latent Diffusion Models.", spec: "Latent Diffusion Model with a VAE encoder, Cross-Attention UNet for denoising, and VAE decoder." },
    { id: "vae", category: "Generative", title: "VAE", author: "Kingma & Welling (2013)", desc: "Auto-Encoding Variational Bayes.", spec: "Variational Autoencoder with Conv2D encoder, reparameterization layer (mu and log_var), and ConvTranspose2D decoder." },
    { id: "gan", category: "Generative", title: "DCGAN", author: "Radford et al. (2015)", desc: "Deep Convolutional Generative Adversarial Networks.", spec: "DCGAN Generator with ConvTranspose2D and BatchNorm, connected to a Discriminator with strided Conv2D and LeakyReLU." },

    // --- State-of-the-Art / Systems ---
    { id: "mamba", category: "SOTA", title: "Mamba", author: "Gu & Dao (2023)", desc: "Linear-Time Sequence Modeling with Selective State Spaces.", spec: "Mamba architecture with Selective State Space Models (SSM), linear scaling, and no attention mechanism." },
    { id: "flashattn", category: "SOTA", title: "FlashAttention", author: "Dao et al. (2022)", desc: "Fast and Memory-Efficient Exact Attention with IO-Awareness.", spec: "Transformer block utilizing FlashAttention for IO-aware, memory-efficient exact self-attention." }
];

function initGallery() {
    const grid = document.getElementById('gallery-grid');
    grid.innerHTML = ''; // clear
    
    referenceLibrary.forEach((paper, index) => {
        const card = document.createElement('div');
        card.className = 'glass-card paper-card';
        // Add a stagger effect to the entrance animation
        card.style.animationDelay = `${index * 0.05}s`;
        card.style.animation = `fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) ${index * 0.05}s both`;

        card.innerHTML = `
            <div class="card-badge badge-${paper.category}">${paper.category}</div>
            <h3>${paper.title}</h3>
            <p class="author">${paper.author}</p>
            <p class="desc">${paper.desc}</p>
            <button class="btn secondary-btn" onclick="loadExample('${paper.id}')">Load in Sandbox</button>
        `;
        grid.appendChild(card);
    });
}

function loadExample(id) {
    switchTab('sandbox');
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if(btn.innerText === "Sandbox") btn.classList.add('active');
    });

    const paper = referenceLibrary.find(p => p.id === id);
    document.getElementById('text-a').value = paper.spec;
    document.getElementById('text-b').value = ""; // Clear comparison
    analyzeArchitecture();
}

// ── Interactive Architect Engine ──────────────────────────

let currentBlueprint = []; // List of {type, params}
let architectTarget = 'architect'; // 'architect' or 'editor'

const LAYER_TYPES = [
    "Conv2D", "Linear", "MultiHeadAttention", "TransformerBlock", 
    "LayerNorm", "BatchNorm2D", "ReLU", "GELU", "Dropout", 
    "ResBlock", "PatchEmbedding", "SequencePooling", "Flatten",
    "CausalAttention", "CrossAttention", "ResidualAdd", "FeedForward"
];

const PARAM_DEFAULTS = {
    "Conv2D": { channels: 64, kernel_size: 3, stride: 1 },
    "Linear": { hidden_size: 768 },
    "MultiHeadAttention": { embed_dim: 768, num_heads: 12 },
    "TransformerBlock": { embed_dim: 768, num_heads: 12 },
    "PatchEmbedding": { patch_size: 16, embed_dim: 768 },
    "ResBlock": { channels: 64 },
    "FeedForward": { embed_dim: 768, ff_dim: 3072 },
    "LayerNorm": { normalized_shape: 768 },
    "BatchNorm2D": { num_features: 64 },
    "Dropout": { p: 0.1 },
    "CausalAttention": { embed_dim: 768, num_heads: 12 },
    "CrossAttention": { embed_dim: 768, num_heads: 12 },
    "SequencePooling": { mode: "mean" }
};

const CHALLENGES = [
    {
        name: "The Residual Mismatch",
        goal: "Fix the residual connection. The main path currently outputs 128 channels, but the identity shortcut only has 64. Add a projection layer or fix the dimensions.",
        initialLayers: [
            { type: "Conv2D", params: { channels: 128, kernel_size: 3 } },
            { type: "residual_add", params: { label: "Broken Shortcut" } }
        ],
        targetMotifs: ["ResBlock"],
        successCriteria: (data) => !data.metadata.failure && data.kag_motifs.includes("ResBlock")
    },
    {
        name: "Transformer Head Repair",
        goal: "Correct the MultiHeadAttention configuration. The embedding dimension is 768, but num_heads is set to 10 (not divisible). Fix it to a valid divisor.",
        initialLayers: [
            { type: "MultiHeadAttention", params: { embed_dim: 768, num_heads: 10 } }
        ],
        successCriteria: (data) => !data.metadata.failure && data.tensor_trace.length > 0
    },
    {
        name: "Memory Optimization",
        goal: "Reduce the activation memory of this simple stack. The current Conv2D with 512 channels is too heavy. Use a bottleneck or reduce channels.",
        initialLayers: [
            { type: "Conv2D", params: { channels: 512, kernel_size: 3 } },
            { type: "Conv2D", params: { channels: 512, kernel_size: 3 } }
        ],
        successCriteria: (data) => {
            const mem = data.flops_events.reduce((acc, e) => acc + (e.mem_mb || 0), 0);
            return mem < 50; // Arbitrary memory target
        }
    },
    {
        name: "Cross-Attention Routing",
        goal: "Build a valid Encoder-Decoder bridge. Connect an Encoder (768 dim) to a CrossAttention layer. Ensure dimensions match.",
        initialLayers: [
            { type: "Linear", params: { hidden_size: 512, label: "Encoder Output" } },
            { type: "CrossAttention", params: { embed_dim: 768, num_heads: 12 } }
        ],
        successCriteria: (data) => !data.metadata.failure && data.kag_semantic_roles && Object.values(data.kag_semantic_roles).includes("cross_attention")
    }
];

let activeChallengeIndex = -1;

function initArchitect(initialLayers = []) {
    currentBlueprint = initialLayers.length > 0 ? initialLayers : [
        { type: "PatchEmbedding", params: { patch_size: 16, embed_dim: 768 } },
        { type: "MultiHeadAttention", params: { embed_dim: 768, num_heads: 12 } },
        { type: "FeedForward", params: { embed_dim: 768, ff_dim: 3072 } }
    ];
    renderEditorLayers();
    updateArchitecture();
}

function renderEditorLayers() {
    const listId = architectTarget === 'architect' ? 'architect-layer-list' : 'editor-layer-list';
    const container = document.getElementById(listId);
    if (!container) return;
    
    container.innerHTML = '';
    
    currentBlueprint.forEach((layer, idx) => {
        const div = document.createElement('div');
        div.className = 'editor-layer-item';
        div.innerHTML = `
            <div class="layer-drag-handle">⠿</div>
            <div class="layer-controls">
                <select class="layer-type-select" onchange="changeLayerType(${idx}, this.value)">
                    ${LAYER_TYPES.map(t => `<option value="${t}" ${t.toLowerCase() === layer.type.toLowerCase() ? 'selected' : ''}>${t}</option>`).join('')}
                </select>
                <button class="layer-remove-btn" onclick="removeEditorLayer(${idx})">×</button>
            </div>
            <div class="layer-props">
                ${Object.entries(layer.params).map(([k, v]) => `
                    <div class="prop-group">
                        <label>${k}</label>
                        <input type="text" class="prop-input" value="${v}" onchange="changeLayerParam(${idx}, '${k}', this.value)">
                    </div>
                `).join('')}
            </div>
        `;
        container.appendChild(div);
    });
}

function addEditorLayer() {
    const newLayer = { type: "Linear", params: { ...PARAM_DEFAULTS["Linear"] } };
    currentBlueprint.push(newLayer);
    renderEditorLayers();
    updateArchitecture();
}

function removeEditorLayer(idx) {
    currentBlueprint.splice(idx, 1);
    renderEditorLayers();
    updateArchitecture();
}

function changeLayerType(idx, newType) {
    currentBlueprint[idx].type = newType;
    currentBlueprint[idx].params = { ...(PARAM_DEFAULTS[newType] || {}) };
    renderEditorLayers();
    updateArchitecture();
}

function changeLayerParam(idx, key, value) {
    const numVal = parseInt(value);
    currentBlueprint[idx].params[key] = isNaN(numVal) ? value : numVal;
    updateArchitecture();
}

async function updateArchitecture() {
    const badgeId = architectTarget === 'architect' ? 'architect-status-badge' : 'editor-status-badge';
    const consoleId = architectTarget === 'architect' ? 'architect-console' : 'editor-console';
    const logId = architectTarget === 'architect' ? 'architect-validation-log' : 'editor-validation-log';
    
    const badge = document.getElementById(badgeId);
    const console = document.getElementById(consoleId);
    const log = document.getElementById(logId);
    
    if (badge) { badge.className = 'status-badge running'; badge.innerText = 'Running...'; }
    
    try {
        const response = await fetch('/api/analyze_graph', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                name: "Architect Session",
                layers: currentBlueprint 
            })
        });
        
        const data = await response.json();
        
        if (badge) {
            const hasError = data.kag_anomalies && data.kag_anomalies.length > 0;
            badge.className = `status-badge ${hasError ? 'invalid' : 'valid'}`;
            badge.innerText = hasError ? 'Invalid Flow' : 'Valid Architecture';
        }
        
        // Render Console (Tensor Trace)
        renderArchitectConsole(data.tensor_trace || [], console);
        
        // Render Validation Log
        renderArchitectValidation(data, log);
        
        // Sync with global results if needed
        if (architectTarget === 'editor') {
            renderResults(data);
        } else {
            renderResults(data);
        }
        
        // --- Failure Localization Banner ---
        const existingBanner = document.getElementById('failure-localization-banner');
        if (existingBanner) existingBanner.remove();

        if (data.metadata && data.metadata.failure) {
            const f = data.metadata.failure;
            const banner = document.createElement('div');
            banner.id = 'failure-localization-banner';
            banner.className = 'glass-panel';
            banner.style.borderLeft = '4px solid #ef4444';
            banner.style.background = 'rgba(239, 68, 68, 0.1)';
            banner.style.padding = '1.5rem';
            banner.style.marginTop = '1.5rem';
            banner.style.marginBottom = '1.5rem';
            
            banner.innerHTML = `
                <div style="display: flex; gap: 15px; align-items: flex-start;">
                    <div style="font-size: 2rem;">🛑</div>
                    <div>
                        <h4 style="color: #fca5a5; margin-bottom: 5px;">Tensor Failure Localized</h4>
                        <p style="color: #f87171; font-family: monospace; font-size: 0.95rem; margin-bottom: 10px;">${f.message}</p>
                        <p style="color: #94a3b8; font-size: 0.85rem;">
                            <strong>Localization:</strong> Break occurred at node <code>${f.node_id}</code>. 
                            The upstream dependency chain of ${f.upstream_path.length} nodes has been highlighted in yellow.
                        </p>
                    </div>
                </div>
            `;
            // Insert before results area
            const resArea = document.getElementById('results-area');
            resArea.parentNode.insertBefore(banner, resArea);
            
            // Highlight in Graph (if graph exists)
            highlightFailureInGraph(f.node_id, f.upstream_path);
        }

    } catch (err) {
        if (badge) { badge.className = 'status-badge invalid'; badge.innerText = 'Error'; }
        if (console) console.innerHTML = `<div style="color:#ef4444;">API Error: ${err.message}</div>`;
    }
}

function renderArchitectConsole(trace, container) {
    if (!container) return;
    container.innerHTML = '';
    if (trace.length === 0) {
        container.innerHTML = '<div class="console-placeholder">Awaiting propagation...</div>';
        return;
    }
    
    trace.forEach(line => {
        const div = document.createElement('div');
        div.style.marginBottom = '8px';
        div.style.borderLeft = '2px solid #3b82f6';
        div.style.paddingLeft = '10px';
        
        // Highlight shapes
        const html = line.replace(/(\(.*?\))/g, '<span style="color:#34d399; font-weight:bold;">$1</span>')
                         .replace(/\[Trace\]/g, '<span style="color:#94a3b8;">[Trace]</span>');
        div.innerHTML = html;
        container.appendChild(div);
    });
}

function renderArchitectValidation(data, container) {
    if (!container) return;
    container.innerHTML = '';
    
    const events = [];
    
    // Add anomalies
    (data.kag_anomalies || []).forEach(a => {
        events.push({ type: 'error', text: a });
    });
    
    // Add motifs as successes
    (data.kag_motifs || []).forEach(m => {
        events.push({ type: 'success', text: `Motif recognized: ${m}` });
    });
    
    // Add FLOPs warnings
    (data.flops_events || []).forEach(f => {
        (f.warnings || []).forEach(w => {
            events.push({ type: 'warning', text: `${f.node_id}: ${w}` });
        });
    });

    if (events.length === 0) {
        container.innerHTML = '<div style="color:#475569; font-style:italic;">No critical topology events detected.</div>';
        return;
    }
    
    events.forEach(ev => {
        const div = document.createElement('div');
        div.className = `val-log-item ${ev.type}`;
        const icon = ev.type === 'success' ? '✅' : ev.type === 'error' ? '❌' : '⚠️';
        div.innerHTML = `<span>${icon}</span> <span>${ev.text}</span>`;
        container.appendChild(div);
    });
}

// Hook into the tab switcher to initialize editor
const originalSwitchTab = switchTab;
switchTab = function(tabId, event) {
    originalSwitchTab(tabId, event);
    if (tabId === 'architect-view') {
        architectTarget = 'architect';
        initArchitect();
    }
};

const originalSwitchResultTab = switchResultTab;
switchResultTab = function(tabId, event) {
    originalSwitchResultTab(tabId, event);
    if (tabId === 'editor-view') {
        architectTarget = 'editor';
        // Initialize with current result's layers
        const currentLayers = lastResultData ? lastResultData.layer_breakdown.map(l => ({
            type: l.type,
            params: l.params
        })) : [];
        initArchitect(currentLayers);
    }
};

let lastResultData = null;
const originalRenderResults = renderResults;
renderResults = function(data, isComparison) {
function highlightFailureInGraph(failingNodeId, upstreamPath) {
    // Wait a bit for Viz to finish rendering
    setTimeout(() => {
        const svg = document.querySelector('#graph-a svg');
        if (!svg) return;
        
        // Find all nodes (usually titles or IDs are in <title> tags inside <g class="node">)
        const nodes = svg.querySelectorAll('.node');
        nodes.forEach(node => {
            const title = node.querySelector('title').textContent;
            
            if (title === failingNodeId) {
                // Highlight failing node
                const polygon = node.querySelector('polygon') || node.querySelector('ellipse') || node.querySelector('path');
                if (polygon) {
                    polygon.style.stroke = '#ef4444';
                    polygon.style.strokeWidth = '4px';
                    polygon.style.fill = 'rgba(239, 68, 68, 0.3)';
                }
            } else if (upstreamPath.includes(title)) {
                // Highlight upstream dependency
                const polygon = node.querySelector('polygon') || node.querySelector('ellipse') || node.querySelector('path');
                if (polygon) {
                    polygon.style.stroke = '#fbbf24';
                    polygon.style.strokeWidth = '3px';
                    polygon.style.fill = 'rgba(245, 158, 11, 0.1)';
                }
            }
        });
        
        // Optionally highlight edges between these nodes
        const edges = svg.querySelectorAll('.edge');
        edges.forEach(edge => {
            const title = edge.querySelector('title').textContent;
            // Edge title is usually "source->target"
            const [src, tgt] = title.split('->');
            if (upstreamPath.includes(src) && (upstreamPath.includes(tgt) || tgt === failingNodeId)) {
                const path = edge.querySelector('path');
                if (path) {
                    path.style.stroke = '#f59e0b';
                    path.style.strokeWidth = '2px';
                }
                const polygon = edge.querySelector('polygon');
                if (polygon) {
                    polygon.style.fill = '#f59e0b';
                    polygon.style.stroke = '#f59e0b';
                }
            }
        });
    }, 500);
}

// Update renderResults to handle failures from any source
const originalRenderResultsFailureCheck = renderResults;
renderResults = function(data, isComparison) {
    // Clear failure banner if it exists
    const existingBanner = document.getElementById('failure-localization-banner');
    if (existingBanner) existingBanner.remove();

    if (data.metadata && data.metadata.failure) {
        // We'll let the architect code handle it for now or move it here
        // For consistency, let's move it here:
        const f = data.metadata.failure;
        const banner = document.createElement('div');
        banner.id = 'failure-localization-banner';
        banner.className = 'glass-panel';
        banner.style.borderLeft = '4px solid #ef4444';
        banner.style.background = 'rgba(239, 68, 68, 0.1)';
        banner.style.padding = '1.5rem';
        banner.style.marginTop = '1.5rem';
        banner.style.marginBottom = '1.5rem';
        
        banner.innerHTML = `
            <div style="display: flex; gap: 15px; align-items: flex-start;">
                <div style="font-size: 2rem;">🛑</div>
                <div>
                    <h4 style="color: #fca5a5; margin-bottom: 5px;">Tensor Failure Localized</h4>
                    <p style="color: #f87171; font-family: monospace; font-size: 0.95rem; margin-bottom: 10px;">${f.message}</p>
                    <p style="color: #94a3b8; font-size: 0.85rem;">
                        <strong>Localization:</strong> Break occurred at node <code>${f.node_id}</code>. 
                        The upstream dependency chain of ${f.upstream_path.length} nodes has been highlighted in yellow.
                    </p>
                </div>
            </div>
        `;
        const resArea = document.getElementById('results-area');
        if (resArea) resArea.parentNode.insertBefore(banner, resArea);
        
        highlightFailureInGraph(f.node_id, f.upstream_path);
    }
    
    originalRenderResultsFailureCheck(data, isComparison);
};

function renderComparisonDiff(diff) {
    const diffPanel = document.getElementById('comparison-diff-panel');
    const metricsList = document.getElementById('diff-metrics-list');
    const topologyList = document.getElementById('diff-topology-list');
    const summaryBox = document.getElementById('diff-semantic-summary');
    
    if (!diffPanel) return;
    diffPanel.style.display = 'block';
    
    // 1. Metrics
    metricsList.innerHTML = '';
    const metricLabels = {
        'flops': 'FLOPs Score',
        'params': 'Parameter Count',
        'depth': 'Arch Depth',
        'memory': 'Activation VRAM'
    };
    
    for (const [key, val] of Object.entries(diff.deltas)) {
        const div = document.createElement('div');
        div.style.display = 'flex';
        div.style.justifyContent = 'space-between';
        div.style.fontSize = '0.85rem';
        
        const label = metricLabels[key] || key;
        const sign = val >= 0 ? '+' : '';
        const color = val > 0 ? '#f87171' : val < 0 ? '#34d399' : '#94a3b8';
        
        // Formatting
        let displayVal = val;
        if (key === 'params' || key === 'flops') {
            if (Math.abs(val) > 1000000) displayVal = (val / 1000000).toFixed(1) + 'M';
            else if (Math.abs(val) > 1000) displayVal = (val / 1000).toFixed(1) + 'K';
        } else if (key === 'memory') {
            displayVal = val.toFixed(1) + ' MB';
        }

        div.innerHTML = `
            <span>${label}</span>
            <span style="color: ${color}; font-weight: bold;">${sign}${displayVal}</span>
        `;
        metricsList.appendChild(div);
    }
    
    // 2. Topology
    topologyList.innerHTML = '';
    const changes = [];
    
    diff.added_nodes.forEach(n => changes.push({ type: 'add', text: `Added **${n}**` }));
    diff.removed_nodes.forEach(n => changes.push({ type: 'remove', text: `Removed **${n}**` }));
    diff.changed_params.forEach(c => {
        let desc = `Modified **${c.label}**`;
        if (c.type_changed) desc += ` (Type changed to ${c.to.type})`;
        changes.push({ type: 'change', text: desc });
    });
    
    if (changes.length === 0) {
        topologyList.innerHTML = '<div style="color:#475569; font-style:italic;">No topological differences detected.</div>';
    } else {
        changes.slice(0, 6).forEach(ch => {
            const div = document.createElement('div');
            div.style.fontSize = '0.8rem';
            div.style.padding = '4px 8px';
            div.style.borderRadius = '4px';
            
            if (ch.type === 'add') {
                div.style.background = 'rgba(16, 185, 129, 0.1)';
                div.style.color = '#34d399';
                div.innerHTML = `+ ${ch.text}`;
            } else if (ch.type === 'remove') {
                div.style.background = 'rgba(239, 68, 68, 0.1)';
                div.style.color = '#f87171';
                div.innerHTML = `- ${ch.text}`;
            } else {
                div.style.background = 'rgba(245, 158, 11, 0.1)';
                div.style.color = '#fbbf24';
                div.innerHTML = `Δ ${ch.text}`;
            }
            topologyList.appendChild(div);
        });
        if (changes.length > 6) {
            const more = document.createElement('div');
            more.style.fontSize = '0.75rem';
            more.style.color = '#94a3b8';
            more.style.marginTop = '4px';
            more.innerText = `+ ${changes.length - 6} more changes...`;
            topologyList.appendChild(more);
        }
    }
    
    // 3. Summary
    summaryBox.innerHTML = `<strong>Semantic Insights:</strong> ${diff.summary}`;
}
async function analyzeArchitecture() {
    const textA = document.getElementById('text-a').value.trim();
    const textB = document.getElementById('text-b').value.trim();

    if (!textA && !textB) {
        alert("Please provide at least Architecture A.");
        return;
    }

    // UI State: Loading
    const btn = document.getElementById('analyze-btn');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    
    btn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'inline-block';

    const resultsArea = document.getElementById('results-area');
    resultsArea.style.display = 'none';

    try {
        const isComparison = textA && textB;
        const endpoint = isComparison ? '/api/compare_text' : '/api/parse_text';
        
        const payload = isComparison 
            ? { text_a: textA, text_b: textB }
            : { text: textA || textB }; // If only B is filled, treat as A

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "API Error");
        }

        const data = await response.json();
        renderResults(data, isComparison);

    } catch (error) {
        alert("Error analyzing architecture: " + error.message);
    } finally {
        // UI State: Restore
        btn.disabled = false;
        btnText.style.display = 'inline-block';
        loader.style.display = 'none';
    }
}

async function analyzePDF() {
    const fileInput = document.getElementById('pdf-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a PDF file to upload.");
        return;
    }

    const btn = document.getElementById('analyze-pdf-btn');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    
    btn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'inline-block';

    const resultsArea = document.getElementById('results-area');
    resultsArea.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/parse_pdf', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "API Error");
        }

        const data = await response.json();
        // Reset tabs to default Graph View
        document.querySelectorAll('.res-nav-btn')[0].click();
        renderResults(data, false);

    } catch (error) {
        alert("Error processing PDF: " + error.message);
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline-block';
        loader.style.display = 'none';
    }
}

function renderResults(data, isComparison) {
    const resultsArea = document.getElementById('results-area');
    const visualsGrid = document.getElementById('visuals-grid');
    const panelA = document.getElementById('panel-a');
    const panelB = document.getElementById('panel-b');
    const graphA = document.getElementById('graph-a');
    const graphB = document.getElementById('graph-b');
    const titleA = document.getElementById('title-a');
    const titleB = document.getElementById('title-b');
    const expContent = document.getElementById('explanation-content');
    const codePanel = document.getElementById('code-panel');
    const codeBlock = document.getElementById('pytorch-code');
    const codeBadge = document.getElementById('code-source-badge');
    const metricsPanel = document.getElementById('metrics-panel');
    const layerList = document.getElementById('layer-list');

    resultsArea.style.display = 'flex';
    
    // Render Markdown
    expContent.innerHTML = marked.parse(data.explanation);

    // Render Layer Explorer
    if (data.layer_breakdown && !isComparison) {
        layerList.innerHTML = '';
        data.layer_breakdown.forEach(layer => {
            let paramsHtml = '';
            for (const [key, val] of Object.entries(layer.params)) {
                if (key.startsWith('_')) continue;
                paramsHtml += `<span class="param-badge">${key}: <strong>${val}</strong></span>`;
            }
            
            // Add semantic params if present
            if (layer.semantic) {
                for (const [key, val] of Object.entries(layer.semantic)) {
                    paramsHtml += `<span class="param-badge" style="border-color: var(--accent);">semantic_${key}: <strong style="color:var(--accent);">${val}</strong></span>`;
                }
            }

            layerList.innerHTML += `
                <div class="layer-item">
                    <div class="layer-header">
                        <span class="layer-id">${layer.id}</span>
                        <span class="layer-type">${layer.type}</span>
                    </div>
                    ${layer.description ? `<div class="layer-desc"><i>"${layer.description}"</i></div>` : ''}
                    
                    <div class="layer-params mb-2">${paramsHtml}</div>
                    
                    <div class="layer-expl mt-2" style="font-size: 0.9rem; color: #a5b4fc;">
                        ${layer.explanation ? marked.parse(layer.explanation) : ''}
                    </div>

                    ${layer.code_snippet ? `
                        <div class="layer-code mt-2">
                            <pre style="margin:0; padding:0.5rem; background:#111; border-radius:4px;"><code class="language-python">${layer.code_snippet}</code></pre>
                        </div>
                    ` : ''}
                </div>
            `;
        });
        
        // Highlight new snippets
        layerList.querySelectorAll('pre code').forEach((el) => {
            hljs.highlightElement(el);
        });
    }

    // Render Metrics (If available - currently only for single arch)
    if (data.metrics && !isComparison) {
        metricsPanel.style.display = 'grid';
        
        document.getElementById('metric-flops').textContent = data.metrics.flops_score.toFixed(1);
        
        let params = data.metrics.params;
        let paramsStr = params > 1e6 ? `~${(params/1e6).toFixed(1)}M` : `~${(params/1e3).toFixed(0)}K`;
        document.getElementById('metric-params').textContent = paramsStr;
        
        document.getElementById('metric-depth').textContent = data.metrics.depth;
        document.getElementById('metric-memory').textContent = `${data.metrics.memory_mb} MB`;
        
        // KAG Status
        const kagEl = document.getElementById('metric-kag');
        if (kagEl) {
            const motifCount = (data.kag_motifs || []).length;
            const anomalyCount = (data.kag_anomalies || []).length;
            if (anomalyCount > 0) {
                kagEl.innerHTML = `<span style="color:#f87171;">⚠️ ${anomalyCount} Anomalies</span>`;
            } else if (motifCount > 0) {
                kagEl.innerHTML = `<span style="color:#34d399;">✅ ${motifCount} Motifs</span>`;
            } else {
                kagEl.innerHTML = `<span style="color:#94a3b8;">⚪ Logical Pass</span>`;
            }
        }
    } else {
        metricsPanel.style.display = 'none';
    }

    // Render Graphs (Using viz.js)
    const viz = new Viz();
    
    if (isComparison) {
        visualsGrid.classList.add('compare');
        panelA.style.display = 'block';
        panelB.style.display = 'block';
        codePanel.style.display = 'none'; // Don't show code in comparison mode yet
        
        titleA.innerText = data.name_a;
        titleB.innerText = data.name_b;

        viz.renderSVGElement(data.svg_a).then(element => {
            graphA.innerHTML = "";
            graphA.appendChild(element);
        });
        viz.renderSVGElement(data.svg_b).then(element => {
            graphB.innerHTML = "";
            graphB.appendChild(element);
        });

        // Populate Comparison Diff
        if (data.comparison_result) {
            renderComparisonDiff(data.comparison_result);
        }
    } else {
        visualsGrid.classList.remove('compare');
        panelA.style.display = 'block';
        panelB.style.display = 'none';
        
        const diffPanel = document.getElementById('comparison-diff-panel');
        if (diffPanel) diffPanel.style.display = 'none';
        
        titleA.innerText = data.name;

        viz.renderSVGElement(data.svg).then(element => {
            graphA.innerHTML = "";
            graphA.appendChild(element);
        });

        // Show PyTorch Code
        codePanel.style.display = 'block';
        codeBlock.textContent = data.code;
        codeBadge.textContent = data.code_source;
        hljs.highlightElement(codeBlock);
        
        // Tensor Trace
        if (data.tensor_trace) {
            renderTensorTrace(data.tensor_trace);
        }
        
        // KAG Reasoning
        if (data.kag_anomalies || data.kag_motifs) {
            renderKAGInsights(data);
        }
        
        // Memory Trace
        if (data.flops_events) {
            renderMemoryTrace(data.flops_events);
        }
    }
}


    // Initialize Tensor Flow
    if (!isComparison && data.tensor_trace) {
        initTensorFlow(
            data.tensor_trace,
            data.cross_attention_events || [],
            data.flops_events           || []
        );
    }

    // Enrich Layer Explorer with FLOPs data
    if (!isComparison && data.flops_events && data.flops_events.length > 0) {
        enrichLayerExplorerWithFlops(data.flops_events);
    }

    // Initialize KAG Reasoning Panel
    if (!isComparison) {
        renderKagReasoning(
            data.kag_motifs   || [],
            data.kag_anomalies || [],
            data.kag_semantic_roles || {}
        );
    }
}

// --- Tensor Flow View ---
let currentTensorStep = 0;
let tensorTraceData   = [];
let tensorCrossEvents = [];
let tensorFlopsMap    = {};   // nodeId → FLOPsResult dict
let tensorPlayInterval = null;

function initTensorFlow(trace, crossEvents, flopsEvents) {
    tensorTraceData   = trace;
    tensorCrossEvents = crossEvents || [];

    // Build quick-lookup map keyed by node_id
    tensorFlopsMap = {};
    (flopsEvents || []).forEach(ev => { tensorFlopsMap[ev.node_id] = ev; });

    currentTensorStep = 0;
    renderTensorStep();

    // FLOPs overlay bar
    renderFlopsOverlay(flopsEvents || []);

    // Cross-attention panel only when there are events
    const caPanel = document.getElementById('cross-attn-panel');
    if (caPanel) {
        caPanel.style.display = tensorCrossEvents.length > 0 ? 'block' : 'none';
        if (tensorCrossEvents.length > 0) renderCrossAttentionPanel(tensorCrossEvents);
    }
}

function parseTensorTrace(line) {
    // Expected format: "[Trace] node_id (node_type): in_shape -> out_shape"
    const match = line.match(/\[Trace\] (.*?) \((.*?)\): (.*?) -> (.*)/);
    if (match) {
        return {
            id: match[1],
            type: match[2],
            in: match[3],
            out: match[4]
        };
    }
    return null;
}

function renderTensorStep() {
    const container = document.getElementById('tensor-flow-container');
    container.innerHTML = '';
    
    if (!tensorTraceData || tensorTraceData.length === 0) {
        container.innerHTML = '<div style="color:var(--text-secondary);">No tensor trace available.</div>';
        return;
    }

    const SEVERITY_COLORS = {
        critical: { bg: 'rgba(239,68,68,0.15)',  border: '#ef4444', badge: '#fca5a5' },
        high:     { bg: 'rgba(245,158,11,0.15)', border: '#f59e0b', badge: '#fcd34d' },
        medium:   { bg: 'rgba(99,102,241,0.12)', border: '#818cf8', badge: '#a5b4fc' },
        low:      { bg: 'rgba(255,255,255,0.04)', border: 'transparent', badge: '#64748b' },
    };
    
    // Render up to currentStep
    for (let i = 0; i <= currentTensorStep && i < tensorTraceData.length; i++) {
        const stepData = parseTensorTrace(tensorTraceData[i]);
        if (!stepData) continue;

        const flops = tensorFlopsMap[stepData.id] || null;
        const severity = flops ? flops.severity : 'low';
        const sc = SEVERITY_COLORS[severity];
        const isCurrent = (i === currentTensorStep);

        const div = document.createElement('div');
        div.style.cssText = `
            padding: 10px 14px; border-radius: 8px;
            background: ${isCurrent ? 'rgba(59,130,246,0.18)' : sc.bg};
            border-left: 4px solid ${isCurrent ? '#3b82f6' : sc.border};
            display: flex; flex-direction: column; gap: 6px;
            transition: all 0.3s;
        `;

        // Semantic tags
        let tags = '';
        if (stepData.type === 'multiheadattention' || stepData.type === 'mhsa')
            tags += '<span class="flops-tag tag-attn">🧠 Attention</span>';
        if (stepData.type === 'cross_attention')
            tags += '<span class="flops-tag tag-cross">⚡ Cross-Attn</span>';
        if (stepData.type === 'causal_attention')
            tags += '<span class="flops-tag tag-causal">🔒 Causal</span>';
        if (stepData.type === 'residual_add')
            tags += '<span class="flops-tag tag-residual">🔗 Residual</span>';
        if (stepData.type === 'feedforward')
            tags += '<span class="flops-tag tag-ff">⚙️ FFN</span>';
        if (severity === 'critical')
            tags += '<span class="flops-tag tag-critical">🔥 Critical</span>';
        else if (severity === 'high')
            tags += '<span class="flops-tag tag-high">⚠️ High Cost</span>';

        // FLOPs row
        let flopsHtml = '';
        if (flops) {
            const mf = flops.flops_mflops.toFixed(2);
            const mm = flops.memory_mb.toFixed(2);
            const maxFlops = Math.max(...Object.values(tensorFlopsMap).map(f => f.flops_mflops), 1);
            const barPct   = Math.min(100, (flops.flops_mflops / maxFlops) * 100).toFixed(1);
            const warnings = flops.warnings.map(w =>
                `<div class="flops-warning">⚠ ${w}</div>`).join('');

            flopsHtml = `
                <div class="flops-row">
                    <div class="flops-bar-wrap">
                        <div class="flops-bar" style="width:${barPct}%; background:${sc.badge};"></div>
                    </div>
                    <div class="flops-meta">
                        <span class="flops-num">${mf} MFLOPs</span>
                        <span class="flops-mem">${mm} MB</span>
                        <code class="flops-complexity">${flops.complexity}</code>
                    </div>
                </div>
                ${flops.formula ? `<div class="flops-formula">${flops.formula.replace(/\n/g,'<br>')}</div>` : ''}
                ${warnings}
            `;
        }

        div.innerHTML = `
            <div style="font-weight:700; color:var(--text-primary); display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
                <span>${stepData.id}</span>
                <span style="color:var(--text-secondary); font-size:0.85em;">(${stepData.type})</span>
                ${tags}
            </div>
            <div style="display:flex; align-items:center; gap:10px; color:#a9b7c6; font-family:monospace; font-size:0.9rem;">
                <span>${stepData.in}</span>
                <span style="color:#3b82f6;">→</span>
                <span style="color:#10b981;">${stepData.out}</span>
            </div>
            ${flopsHtml}
        `;
        
        container.appendChild(div);
        if (isCurrent) div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

function nextTensorStep() {
    if (tensorTraceData && currentTensorStep < tensorTraceData.length - 1) {
        currentTensorStep++;
        renderTensorStep();
    } else {
        pauseTensorFlow();
    }
}

function prevTensorStep() {
    if (currentTensorStep > 0) {
        currentTensorStep--;
        renderTensorStep();
    }
}

function playTensorFlow() {
    if (tensorPlayInterval) {
        pauseTensorFlow();
    } else {
        if (currentTensorStep >= tensorTraceData.length - 1) {
            currentTensorStep = 0; // Restart if at end
        }
        tensorPlayInterval = setInterval(nextTensorStep, 800);
    }
}

function pauseTensorFlow() {
    if (tensorPlayInterval) {
        clearInterval(tensorPlayInterval);
        tensorPlayInterval = null;
    }
}

// --- FLOPs Overlay Summary ---

function renderFlopsOverlay(events) {
    const panel = document.getElementById('flops-summary-panel');
    if (!panel || events.length === 0) { if(panel) panel.style.display='none'; return; }
    panel.style.display = 'block';

    const total  = events.reduce((s, e) => s + e.flops_mflops, 0);
    const totalM = events.reduce((s, e) => s + e.memory_mb, 0);
    const critN  = events.filter(e => e.severity === 'critical').length;
    const highN  = events.filter(e => e.severity === 'high').length;
    const maxF   = Math.max(...events.map(e => e.flops_mflops), 0.001);

    // Summary header
    document.getElementById('flops-total').textContent  = total.toFixed(1) + ' MFLOPs';
    document.getElementById('flops-mem-total').textContent = totalM.toFixed(1) + ' MB';
    document.getElementById('flops-critical').textContent = critN;
    document.getElementById('flops-high').textContent    = highN;

    // Bar chart per layer
    const chart = document.getElementById('flops-chart');
    chart.innerHTML = '';
    events.forEach(ev => {
        const pct = (ev.flops_mflops / maxF * 100).toFixed(1);
        const sColor = { critical:'#ef4444', high:'#f59e0b', medium:'#818cf8', low:'#334155' };
        const col = sColor[ev.severity] || '#334155';
        const bar = document.createElement('div');
        bar.className = 'flops-chart-bar';
        bar.title = `${ev.node_id}: ${ev.flops_mflops.toFixed(2)} MFLOPs\n${ev.complexity}\n${ev.formula}`;
        bar.innerHTML = `
            <div class="flops-chart-fill" style="height:${Math.max(pct,2)}%; background:${col};"></div>
            <div class="flops-chart-label">${ev.node_id.slice(0,8)}</div>
        `;
        chart.appendChild(bar);
    });
}

// --- Layer Explorer FLOPs Enrichment ---

function enrichLayerExplorerWithFlops(flopsEvents) {
    const map = {};
    flopsEvents.forEach(e => { map[e.node_id] = e; });

    document.querySelectorAll('.layer-item').forEach(item => {
        const idEl = item.querySelector('.layer-id');
        if (!idEl) return;
        const nodeId = idEl.textContent.trim();
        const fe = map[nodeId];
        if (!fe) return;

        // Avoid double injection
        if (item.querySelector('.flops-injected')) return;

        const sColor = { critical:'#fca5a5', high:'#fcd34d', medium:'#a5b4fc', low:'#64748b' };
        const col = sColor[fe.severity] || '#64748b';

        const div = document.createElement('div');
        div.className = 'flops-injected';
        div.style.cssText = 'margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; font-size:0.8rem;';
        div.innerHTML = `
            <span style="background:rgba(0,0,0,0.3); border:1px solid ${col}; color:${col}; padding:2px 8px; border-radius:6px; font-family:monospace;">
                ⚡ ${fe.flops_mflops.toFixed(2)} MFLOPs
            </span>
            <span style="background:rgba(0,0,0,0.3); border:1px solid #475569; color:#94a3b8; padding:2px 8px; border-radius:6px; font-family:monospace;">
                💾 ${fe.memory_mb.toFixed(2)} MB
            </span>
            <code style="background:rgba(0,0,0,0.3); border:1px solid #334155; color:#7dd3fc; padding:2px 8px; border-radius:6px;">
                ${fe.complexity}
            </code>
            ${fe.warnings.map(w => `<span style="color:#fbbf24; font-size:0.75rem;">⚠ ${w}</span>`).join('')}
        `;
        item.appendChild(div);
    });
}

// --- Cross-Attention Visualizer ---

function renderCrossAttentionPanel(events) {
    const container = document.getElementById('cross-attn-events');
    if (!container) return;
    container.innerHTML = '';

    events.forEach((ev, idx) => {
        const card = document.createElement('div');
        card.className = 'ca-event-card';
        card.innerHTML = `
            <div class="ca-event-title">
                <span class="ca-badge ca-fusion">⚡ ${ev.semantic.fusion}</span>
                <span style="color: var(--text-secondary); font-size: 0.8rem; font-family: monospace;">${ev.node_id}</span>
            </div>

            <div class="ca-streams">
                <!-- Encoder Stream -->
                <div class="ca-stream ca-encoder-stream">
                    <div class="ca-stream-label ca-badge ca-memory">${ev.semantic.memory}</div>
                    <div class="ca-tensor-box encoder">
                        <div class="ca-tensor-shape">${ev.kv_shape}</div>
                        <div class="ca-tensor-role">K &amp; V matrices</div>
                    </div>
                </div>

                <!-- Score Matrix Diagram -->
                <div class="ca-score-col">
                    <div class="ca-op-label">Q × K<sup>T</sup></div>
                    <div class="ca-score-matrix" id="ca-matrix-${idx}">
                        <div class="ca-score-label">${ev.score_shape}</div>
                        <canvas class="ca-canvas" id="ca-canvas-${idx}" width="100" height="60"></canvas>
                    </div>
                    <div class="ca-op-label" style="margin-top: 4px; font-size: 0.7rem; color: #a5b4fc;">
                        Heads: ${ev.num_heads}
                    </div>
                </div>

                <!-- Decoder Stream -->
                <div class="ca-stream ca-decoder-stream">
                    <div class="ca-stream-label ca-badge ca-query">${ev.semantic.query}</div>
                    <div class="ca-tensor-box decoder">
                        <div class="ca-tensor-shape">${ev.q_shape}</div>
                        <div class="ca-tensor-role">Q matrix</div>
                    </div>
                </div>
            </div>

            <div class="ca-fusion-row">
                <div class="ca-arrow-down">↓</div>
                <div class="ca-tensor-box fusion">
                    <div class="ca-tensor-shape">${ev.out_shape}</div>
                    <div class="ca-tensor-role">${ev.semantic.fusion} Output</div>
                </div>
            </div>
        `;
        container.appendChild(card);

        // Animate the score matrix canvas
        setTimeout(() => animateScoreMatrix(`ca-canvas-${idx}`), 200 + idx * 150);
    });
}

function animateScoreMatrix(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const rows = 6, cols = 8;
    const cw = W / cols, ch = H / rows;

    let frame = 0;
    const totalFrames = 40;

    function draw() {
        ctx.clearRect(0, 0, W, H);
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                // Each cell lights up progressively left-to-right, top-to-bottom
                const cellIdx = r * cols + c;
                const threshold = (frame / totalFrames) * rows * cols;
                const intensity = Math.max(0, Math.min(1, (threshold - cellIdx) / 3));

                // Colour: blue (cold) → green (hot attention)
                const r_val = Math.round(16  + intensity * 59);
                const g_val = Math.round(185 * intensity);
                const b_val = Math.round(129 * intensity);
                ctx.fillStyle = `rgba(${r_val}, ${g_val}, ${b_val}, ${0.15 + intensity * 0.75})`;
                ctx.fillRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
            }
        }
        if (frame < totalFrames) {
            frame++;
            requestAnimationFrame(draw);
        }
    }
    draw();
}

// --- KAG Reasoning Panel ---

const KAG_ROLE_COLORS = {
    patch_embedding:    { bg: 'rgba(59,130,246,0.15)',  border: 'rgba(59,130,246,0.4)',  text: '#60a5fa' },
    token_mixer:        { bg: 'rgba(16,185,129,0.15)',  border: 'rgba(16,185,129,0.4)',  text: '#34d399' },
    sequence_encoder:   { bg: 'rgba(168,85,247,0.15)', border: 'rgba(168,85,247,0.4)', text: '#c084fc' },
    feature_aggregator: { bg: 'rgba(245,158,11,0.15)', border: 'rgba(245,158,11,0.4)', text: '#fbbf24' },
    classifier_head:    { bg: 'rgba(239,68,68,0.15)',  border: 'rgba(239,68,68,0.4)',  text: '#f87171' },
    encoder:            { bg: 'rgba(6,182,212,0.15)',   border: 'rgba(6,182,212,0.4)',   text: '#22d3ee' },
    decoder:            { bg: 'rgba(236,72,153,0.15)', border: 'rgba(236,72,153,0.4)', text: '#f472b6' },
};

function renderKagReasoning(motifs, anomalies, roles) {
    // --- Motifs ---
    const motifList = document.getElementById('kag-motifs-list');
    motifList.innerHTML = '';
    if (motifs.length === 0) {
        motifList.innerHTML = '<li style="color: var(--text-secondary); font-size: 0.9rem;">No known motifs detected.</li>';
    } else {
        motifs.forEach(m => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span style="
                    display: inline-flex; align-items: center; gap: 6px;
                    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
                    color: #34d399; padding: 6px 12px; border-radius: 999px;
                    font-size: 0.85rem; font-weight: 600;
                ">✅ ${m}</span>
            `;
            motifList.appendChild(li);
        });
    }

    // --- Anomalies ---
    const anomalyList = document.getElementById('kag-anomalies-list');
    anomalyList.innerHTML = '';
    if (anomalies.length === 0) {
        anomalyList.innerHTML = '<li style="color: #34d399; font-size: 0.9rem;">✅ No topology anomalies detected.</li>';
    } else {
        anomalies.forEach(a => {
            const isIncompat = a.toLowerCase().includes('incompatible');
            const color = isIncompat ? '#f87171' : '#fbbf24';
            const borderColor = isIncompat ? 'rgba(239,68,68,0.3)' : 'rgba(245,158,11,0.3)';
            const bgColor = isIncompat ? 'rgba(239,68,68,0.08)' : 'rgba(245,158,11,0.08)';
            const li = document.createElement('li');
            li.innerHTML = `
                <div style="
                    background: ${bgColor}; border-left: 3px solid ${color};
                    border: 1px solid ${borderColor}; border-radius: 6px;
                    padding: 8px 12px; color: ${color}; font-size: 0.85rem; line-height: 1.4;
                ">${a}</div>
            `;
            anomalyList.appendChild(li);
        });
    }

    // --- Semantic Role Map ---
    const rolesGrid = document.getElementById('kag-roles-grid');
    rolesGrid.innerHTML = '';
    const entries = Object.entries(roles);
    if (entries.length === 0) {
        rolesGrid.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.9rem;">No semantic roles resolved.</span>';
    } else {
        entries.forEach(([nodeId, role]) => {
            const c = KAG_ROLE_COLORS[role] || { bg: 'rgba(255,255,255,0.05)', border: 'rgba(255,255,255,0.15)', text: '#94a3b8' };
            const chip = document.createElement('div');
            chip.title = `Node: ${nodeId} → Role: ${role}`;
            chip.innerHTML = `
                <div style="
                    background: ${c.bg}; border: 1px solid ${c.border};
                    border-radius: 8px; padding: 5px 10px;
                    display: flex; flex-direction: column; gap: 2px;
                    cursor: default;
                ">
                    <span style="font-size: 0.7rem; color: var(--text-secondary); font-family: monospace;">${nodeId}</span>
                    <span style="font-size: 0.8rem; font-weight: 700; color: ${c.text};">${role.replace(/_/g, ' ')}</span>
                </div>
            `;
            rolesGrid.appendChild(chip);
        });
    }
}

// --- Validation Showcase ---

const validationScenarios = [
    {
        id: "valid_flow",
        title: "Valid Transformer Flow",
        status: "valid",
        explanation: "This scenario demonstrates a perfect standard Vision Transformer topology. The structural alignment of embedding sequence sizes to the Multi-Head Self-Attention layers maps without spatial data loss.",
        error: null,
        logs: `[Trace] patch_emb (patchembedding): ('B', 3, 224, 224) -> ('B', 196, 768)
[Trace] pos_emb (positionalembedding): ('B', 196, 768) -> ('B', 196, 768)
[Trace] mhsa_1 (multiheadattention): ('B', 196, 768) -> ('B', 196, 768)
[Trace] mlp_1 (feedforward): ('B', 196, 768) -> ('B', 196, 768)
[Trace] pool (sequence_pooling): ('B', 196, 768) -> ('B', 768)
[Trace] head (linear): ('B', 768) -> ('B', 1000)
Status: Mathematical Flow OK.`,
        dotStr: `digraph G {
rankdir=TB;
node [style="filled", fontname="Arial", shape="box", color="#475569", fontcolor="white"];
edge [color="#94a3b8"];
patch_emb [label="Patch Embed\n(B, 196, 768)", fillcolor="#10b981"];
pos_emb [label="Pos Embed\n(B, 196, 768)", fillcolor="#10b981"];
mhsa_1 [label="MHSA\nHeads: 12", fillcolor="#10b981"];
mlp_1 [label="MLP\nDim: 3072", fillcolor="#10b981"];
pool [label="Pool\n(B, 768)", fillcolor="#10b981"];
head [label="Head\n(B, 1000)", fillcolor="#10b981"];
patch_emb -> pos_emb;
pos_emb -> mhsa_1;
mhsa_1 -> mlp_1;
mlp_1 -> pool;
pool -> head;
}`
    },
    {
        id: "invalid_heads",
        title: "Invalid Attention Head Count",
        status: "invalid",
        explanation: "Attention dimension 768 cannot be evenly divided across 7 heads. `D_H = 768 / 7 = 109.71` which violates the fundamental assumption of PyTorch's `nn.MultiheadAttention` requiring integer divisions.",
        error: "TensorMismatchError: Attention Split Error at mhsa_1: embed_dim (768) must be divisible by num_heads (7).",
        logs: `[Trace] patch_emb (patchembedding): ('B', 3, 224, 224) -> ('B', 196, 768)
[Trace] pos_emb (positionalembedding): ('B', 196, 768) -> ('B', 196, 768)
[ERROR] Topology check failed at 'mhsa_1'`,
        dotStr: `digraph G {
rankdir=TB;
node [style="filled", fontname="Arial", shape="box", color="#475569", fontcolor="white"];
edge [color="#94a3b8"];
patch_emb [label="Patch Embed\n(B, 196, 768)", fillcolor="#10b981"];
pos_emb [label="Pos Embed\n(B, 196, 768)", fillcolor="#10b981"];
mhsa_1 [label="MHSA\nHeads: 7", fillcolor="#ef4444", penwidth=3, color="white"];
mlp_1 [label="MLP\nDim: 3072", fillcolor="#475569"];
patch_emb -> pos_emb;
pos_emb -> mhsa_1;
mhsa_1 -> mlp_1;
}`
    },
    {
        id: "illegal_reshape",
        title: "Illegal Reshape",
        status: "invalid",
        explanation: "Tensor tracking validates absolute volume size logic. Attempting to force an output shape of `('B', 12, 512, 64)` from an input topology measuring `('B', 12, 1024, 64)` yields an elemental mismatch.",
        error: "TensorMismatchError: Reshape Error at node_k: Total elements mismatch. In: 786432, Out: 393216",
        logs: `[Trace] node_q (reshape): ('B', 12, 1024, 64) -> ('B', 12, 1024, 64)
[ERROR] Topology check failed at 'node_k'`,
        dotStr: `digraph G {
rankdir=TB;
node [style="filled", fontname="Arial", shape="box", color="#475569", fontcolor="white"];
edge [color="#94a3b8"];
input [label="Input\n(B, 12, 1024, 64)", fillcolor="#10b981"];
node_k [label="Reshape\n(B, 12, 512, 64)", fillcolor="#ef4444", penwidth=3, color="white"];
input -> node_k;
}`
    },
    {
        id: "residual_mismatch",
        title: "Residual Mismatch",
        status: "invalid",
        explanation: "Skip connections (Residual Additions) are strict algebraic combinations enforcing exact dimensional overlap. Trying to skip a 768-dimensional root trunk into a bottlenecked 1024-dimensional sequence breaks matrix logic.",
        error: "TensorMismatchError: Topology Error at add_1 (residual_add): Cannot merge tensors with different spatial dimensions: ('B', 512, 768) vs ('B', 512, 1024)",
        logs: `[Trace] norm_1 (layernorm): ('B', 512, 768) -> ('B', 512, 768)
[Trace] proj_1 (linear): ('B', 512, 768) -> ('B', 512, 1024)
[ERROR] Topology check failed at 'add_1'`,
        dotStr: `digraph G {
rankdir=TB;
node [style="filled", fontname="Arial", shape="box", color="#475569", fontcolor="white"];
edge [color="#94a3b8"];
norm_1 [label="LayerNorm\n(B, 512, 768)", fillcolor="#10b981"];
proj_1 [label="Linear\n(B, 512, 1024)", fillcolor="#10b981"];
add_1 [label="Residual Add", fillcolor="#ef4444", penwidth=3, color="white"];
norm_1 -> proj_1;
proj_1 -> add_1;
norm_1 -> add_1 [style="dashed", label="Skip (768)"];
}`
    },
    {
        id: "pos_mismatch",
        title: "Positional Embedding Mismatch",
        status: "warning",
        explanation: "The positional embedding dimension specified (512) does not align with the incoming patch projection embedding size (768). Although sequence models sometimes interpolate spatial definitions, mathematical summation fails without an intermediate linear adapter.",
        error: "TensorMismatchError: Positional Embedding Error at pos_emb: Dimension mismatch. Expected dim 512, got 768",
        logs: `[Trace] patch_emb (patchembedding): ('B', 3, 224, 224) -> ('B', 196, 768)
[ERROR] Topology check failed at 'pos_emb'`,
        dotStr: `digraph G {
rankdir=TB;
node [style="filled", fontname="Arial", shape="box", color="#475569", fontcolor="white"];
edge [color="#94a3b8"];
patch_emb [label="Patch Embed\n(B, 196, 768)", fillcolor="#10b981"];
pos_emb [label="Pos Embed\nExpected: 512", fillcolor="#f59e0b", penwidth=3, color="white"];
patch_emb -> pos_emb;
}`
    },
    {
        id: "invalid_cross_attn",
        title: "Invalid Cross-Attention",
        status: "invalid",
        explanation: "Cross-attention maps sequence arrays heterogeneously but requires standard token vector feature embedding limits. Forcing a Decoder Query space of 512 against an Encoder Key/Value space of 768 prevents the generation of an interaction score matrix.",
        error: "TensorMismatchError: Cross-Attention Error at cross_attn: Embed dim mismatch. Q: 512, KV: 768",
        logs: `[Trace] dec_q (linear): ('B', 128, 512) -> ('B', 128, 512)
[Trace] enc_kv (linear): ('B', 1024, 768) -> ('B', 1024, 768)
[ERROR] Topology check failed at 'cross_attn'`,
        dotStr: `digraph G {
rankdir=TB;
node [style="filled", fontname="Arial", shape="box", color="#475569", fontcolor="white"];
edge [color="#94a3b8"];
dec_q [label="Decoder Query\n(B, 128, 512)", fillcolor="#10b981"];
enc_kv [label="Encoder KV\n(B, 1024, 768)", fillcolor="#10b981"];
cross_attn [label="Cross-Attention", fillcolor="#ef4444", penwidth=3, color="white"];
dec_q -> cross_attn;
enc_kv -> cross_attn;
}`
    }
];

function initValidationShowcase() {
    const list = document.getElementById('scenario-list');
    list.innerHTML = '';
    
    validationScenarios.forEach(scen => {
        const li = document.createElement('li');
        li.className = 'scenario-item';
        li.innerHTML = scen.title;
        li.onclick = () => loadScenario(scen.id, li);
        list.appendChild(li);
    });
    
    // Load first by default
    if(list.firstChild) list.firstChild.click();
}

function loadScenario(id, listItemElement) {
    // Update Active Class
    document.querySelectorAll('.scenario-item').forEach(el => el.classList.remove('active'));
    if(listItemElement) listItemElement.classList.add('active');
    
    const scen = validationScenarios.find(s => s.id === id);
    if(!scen) return;
    
    document.getElementById('val-title').textContent = scen.title;
    
    const badge = document.getElementById('val-badge');
    badge.textContent = scen.status.toUpperCase();
    badge.className = 'badge ' + scen.status;
    badge.style.display = 'inline-block';
    
    document.getElementById('val-explanation').innerHTML = marked.parse(scen.explanation);
    
    document.getElementById('val-logs').textContent = scen.logs;
    
    const errBanner = document.getElementById('val-error-banner');
    if (scen.error) {
        errBanner.textContent = scen.error;
        errBanner.style.display = 'block';
    } else {
        errBanner.style.display = 'none';
    }
    
    // Render Graph
    const viz = new Viz();
    viz.renderSVGElement(scen.dotStr).then(element => {
        const container = document.getElementById('val-graph');
        container.innerHTML = '';
        container.appendChild(element);
    }).catch(e => console.error("Viz.js rendering error:", e));
}

// Ensure components run
window.addEventListener('DOMContentLoaded', () => {
    initValidationShowcase();
    updateChallengesList();
    initGallery();
    initArchitect();
});

function updateChallengesList() {
    const list = document.getElementById('challenges-list');
    if (!list) return;
    list.innerHTML = '';
    
    CHALLENGES.forEach((c, i) => {
        const btn = document.createElement('button');
        btn.className = `btn secondary-btn ${activeChallengeIndex === i ? 'active' : ''}`;
        btn.style.width = '100%';
        btn.style.textAlign = 'left';
        btn.style.justifyContent = 'flex-start';
        btn.style.padding = '10px 15px';
        btn.style.display = 'block';
        btn.style.marginBottom = '0.5rem';
        btn.innerHTML = `
            <div style="font-size: 0.85rem; font-weight: 600; color: #fff;">${c.name}</div>
            <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">Scenario ${i+1}</div>
        `;
        btn.onclick = () => loadChallenge(i);
        list.appendChild(btn);
    });
}

function loadChallenge(index) {
    activeChallengeIndex = index;
    const challenge = CHALLENGES[index];
    
    // Reset editor with challenge layers
    architectTarget = 'architect';
    currentBlueprint = JSON.parse(JSON.stringify(challenge.initialLayers));
    
    // Update UI
    updateChallengesList();
    document.getElementById('challenge-active-box').style.display = 'block';
    document.getElementById('challenge-goal-text').innerText = challenge.goal;
    
    initArchitect(currentBlueprint);
    updateArchitecture();
}

function calculateChallengeScore(data) {
    if (activeChallengeIndex === -1) return null;
    const challenge = CHALLENGES[activeChallengeIndex];
    
    let score = 0;
    let reason = "";
    
    // 1. Correctness (50 points)
    if (!data.metadata.failure) {
        score += 50;
        reason = "Architecture is computationally valid.";
    } else {
        reason = "Tensor propagation failed.";
    }
    
    // 2. Mission Success (30 points)
    if (challenge.successCriteria(data)) {
        score += 30;
        reason += " Challenge goal achieved!";
    }
    
    // 3. Efficiency Bonus (20 points)
    const mem = data.flops_events ? data.flops_events.reduce((acc, e) => acc + (e.mem_mb || 0), 0) : 0;
    if (mem < 100) score += 20;
    else if (mem < 500) score += 10;
    
    return { score, reason };
}

function updateChallengeUI(data) {
    const scoreBox = document.getElementById('challenge-active-box');
    if (!scoreBox || activeChallengeIndex === -1) return;
    
    const result = calculateChallengeScore(data);
    const fill = document.getElementById('score-fill');
    const text = document.getElementById('score-text');
    const grade = document.getElementById('score-grade');
    
    fill.style.width = `${result.score}%`;
    text.innerText = `${result.score}/100`;
    document.getElementById('challenge-reason-box').innerText = result.reason;
    
    if (result.score >= 100) { grade.innerText = 'S - Master Architect'; fill.style.background = '#34d399'; }
    else if (result.score >= 80) { grade.innerText = 'A - Expert'; fill.style.background = '#60a5fa'; }
    else if (result.score >= 50) { grade.innerText = 'B - Competent'; fill.style.background = '#fbbf24'; }
    else { grade.innerText = 'C - Needs Repair'; fill.style.background = '#ef4444'; }
}

// Modify updateArchitecture to hook into challenge logic
const originalUpdateArchitecture = updateArchitecture;
updateArchitecture = async function() {
    await originalUpdateArchitecture();
    if (lastResultData && activeChallengeIndex !== -1) {
        updateChallengeUI(lastResultData);
    }
};
