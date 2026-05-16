// Navigation & Tabs
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

function switchInputType(type) {
    document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.input-area').forEach(a => a.classList.remove('active'));
    
    event.currentTarget.classList.add('active');
    document.getElementById(`input-${type}-area`).classList.add('active');
}

function switchResultTab(tabId) {
    document.querySelectorAll('.res-nav-btn').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.result-tab').forEach(a => a.classList.remove('active'));
    
    event.currentTarget.classList.add('active');
    document.getElementById(tabId).classList.add('active');
}

// The "Big 20" Reference Library Data
const referenceLibrary = [
    // --- Computer Vision ---
    { id: "resnet", category: "Vision", title: "ResNet-18", author: "He et al. (2015)", desc: "Deep Residual Learning for Image Recognition. Introduces skip connections.", spec: "ResNet-18 with Conv2D(64), max pooling, 4 residual blocks, and a linear classification head." },
    { id: "unet", category: "Vision", title: "U-Net", author: "Ronneberger et al. (2015)", desc: "Encoder-decoder with spatial skip connections for biomedical segmentation.", spec: "U-Net with encoder path (Conv2D and MaxPool), symmetric decoder path (UpSample), and skip connections between them." },
    { id: "vit", category: "Vision", title: "Vision Transformer", author: "Dosovitskiy et al. (2020)", desc: "An Image is Worth 16x16 Words. Treats image patches as sequence tokens.", spec: "Vision Transformer with patch embedding, 12 attention blocks (quadratic attention), and MLP head." },
    { id: "yolo", category: "Vision", title: "YOLOv3", author: "Redmon et al. (2018)", desc: "Real-Time Object Detection using a single neural network evaluation.", spec: "Darknet-53 backbone with Conv2D layers, residual blocks, and multi-scale detection heads." },
    { id: "convnext", category: "Vision", title: "ConvNeXt", author: "Liu et al. (2022)", desc: "A ConvNet for the 2020s. Modernizes CNNs using Transformer design choices.", spec: "ConvNeXt with 4 stages of depthwise Conv2D blocks, LayerNorm, and GELU activations." },
    { id: "densenet", category: "Vision", title: "DenseNet", author: "Huang et al. (2017)", desc: "Densely Connected Convolutional Networks. Connects each layer to every other layer.", spec: "DenseNet with initial Conv2D, 4 Dense Blocks (with dense skip connections), transition layers, and global average pooling." },
    { id: "efficientnet", category: "Vision", title: "EfficientNet", author: "Tan & Le (2019)", desc: "Rethinking Model Scaling for Convolutional Neural Networks.", spec: "EfficientNet backbone with MBConv blocks, Squeeze-and-Excitation, and Swish activation." },
    { id: "mobilenet", category: "Vision", title: "MobileNetV2", author: "Sandler et al. (2018)", desc: "Inverted Residuals and Linear Bottlenecks for mobile vision.", spec: "MobileNetV2 with inverted residual blocks, depthwise separable Conv2D, and linear bottlenecks." },
    
    // --- NLP / Transformers ---
    { id: "transformer", category: "NLP", title: "Transformer", author: "Vaswani et al. (2017)", desc: "Attention Is All You Need. Replaces RNNs with multi-head self-attention.", spec: "Transformer with 6 encoder blocks and 6 decoder blocks using MultiHeadAttention and FeedForward networks." },
    { id: "bert", category: "NLP", title: "BERT", author: "Devlin et al. (2018)", desc: "Bidirectional Encoder Representations from Transformers.", spec: "BERT Base with 12 Transformer Encoder blocks, 12 attention heads, and hidden size of 768." },
    { id: "gpt2", category: "NLP", title: "GPT-2", author: "Radford et al. (2019)", desc: "Language Models are Unsupervised Multitask Learners.", spec: "GPT-2 with 12 Transformer Decoder blocks, masked self-attention, and layer normalization." },
    { id: "llama", category: "NLP", title: "Llama 2", author: "Touvron et al. (2023)", desc: "Open Foundation and Fine-Tuned Chat Models.", spec: "Llama 2 architecture with RMSNorm, SwiGLU activation, and Rotary Positional Embeddings (RoPE) in Decoder blocks." },
    { id: "roberta", category: "NLP", title: "RoBERTa", author: "Liu et al. (2019)", desc: "A Robustly Optimized BERT Pretraining Approach.", spec: "RoBERTa with 12 Transformer Encoder blocks using dynamic masking and GELU." },
    { id: "t5", category: "NLP", title: "T5", author: "Raffel et al. (2019)", desc: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.", spec: "T5 Encoder-Decoder architecture with relative position embeddings and 12 blocks each." },

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

// Initialize gallery on load
window.addEventListener('DOMContentLoaded', initGallery);

// API Integration
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
    } else {
        visualsGrid.classList.remove('compare');
        panelA.style.display = 'block';
        panelB.style.display = 'none';
        
        titleA.innerText = data.name;

        viz.renderSVGElement(data.svg).then(element => {
            graphA.innerHTML = "";
            graphA.appendChild(element);
        });

        // Render Code
        if (data.code) {
            codePanel.style.display = 'flex';
            codeBlock.textContent = data.code;
            codeBadge.textContent = data.code_source === 'builder' ? 'Library Builder' : (data.code_source === 'llm' ? 'LLM Generated' : 'Skeleton Fallback');
            // Apply highlight.js
            delete codeBlock.dataset.highlighted;
            hljs.highlightElement(codeBlock);
        } else {
            codePanel.style.display = 'none';
        }
    }
}
