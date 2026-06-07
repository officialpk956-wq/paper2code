// --- Architecture Playground Logic ---
window.playgroundState = {
    architecture: 'ResNet',
    config: {
        base_channels: 64,
        stages: 4,
        blocks_per_stage: 2
    },
    debounceTimer: null
};

window.initPlayground = function() {
    renderPlaygroundControls();
    fetchPlaygroundGraph();
};

function renderPlaygroundControls() {
    const controlsDiv = document.getElementById('pg-controls');
    if (!controlsDiv) return;

    const s = window.playgroundState;
    
    let configHtml = '';
    let presetsHtml = '';
    
    if (s.architecture === 'ResNet') {
        presetsHtml = `
            <option value="none">Custom</option>
            <option value="resnet18">ResNet-18</option>
            <option value="resnet34">ResNet-34</option>
        `;
        configHtml = `
            <div class="mb-4">
                <label class="text-sm text-muted uppercase font-semibold mb-2 block">Base Channels: <span id="pg-lbl-ch">${s.config.base_channels}</span></label>
                <input type="range" min="16" max="256" step="16" value="${s.config.base_channels}" class="slider" style="width:100%" oninput="document.getElementById('pg-lbl-ch').innerText=this.value" onchange="updatePlaygroundConfig('base_channels', parseInt(this.value))">
            </div>
            <div class="mb-4">
                <label class="text-sm text-muted uppercase font-semibold mb-2 block">Stages: <span id="pg-lbl-st">${s.config.stages}</span></label>
                <input type="range" min="1" max="5" step="1" value="${s.config.stages}" class="slider" style="width:100%" oninput="document.getElementById('pg-lbl-st').innerText=this.value" onchange="updatePlaygroundConfig('stages', parseInt(this.value))">
            </div>
            <div class="mb-4">
                <label class="text-sm text-muted uppercase font-semibold mb-2 block">Blocks per Stage: <span id="pg-lbl-bl">${s.config.blocks_per_stage}</span></label>
                <input type="range" min="1" max="6" step="1" value="${s.config.blocks_per_stage}" class="slider" style="width:100%" oninput="document.getElementById('pg-lbl-bl').innerText=this.value" onchange="updatePlaygroundConfig('blocks_per_stage', parseInt(this.value))">
            </div>
        `;
    } else if (s.architecture === 'Transformer') {
        presetsHtml = `
            <option value="none">Custom</option>
            <option value="base">ViT-Base</option>
            <option value="large">ViT-Large</option>
        `;
        configHtml = `
            <div class="mb-4">
                <label class="text-sm text-muted uppercase font-semibold mb-2 block">Hidden Size: <span id="pg-lbl-hs">${s.config.hidden_size}</span></label>
                <select style="width:100%; padding:8px; border-radius:4px; background:var(--bg-panel); color:white;" onchange="updatePlaygroundConfig('hidden_size', parseInt(this.value))">
                    ${[128, 256, 384, 512, 768, 1024, 2048].map(v => `<option value="${v}" ${v===s.config.hidden_size?'selected':''}>${v}</option>`).join('')}
                </select>
            </div>
            <div class="mb-4">
                <label class="text-sm text-muted uppercase font-semibold mb-2 block">Heads: <span id="pg-lbl-hd">${s.config.num_heads}</span></label>
                <input type="range" min="1" max="32" step="1" value="${s.config.num_heads}" class="slider" style="width:100%" oninput="document.getElementById('pg-lbl-hd').innerText=this.value" onchange="updatePlaygroundConfig('num_heads', parseInt(this.value))">
            </div>
            <div class="mb-4">
                <label class="text-sm text-muted uppercase font-semibold mb-2 block">Depth: <span id="pg-lbl-dp">${s.config.depth}</span></label>
                <input type="range" min="1" max="12" step="1" value="${s.config.depth}" class="slider" style="width:100%" oninput="document.getElementById('pg-lbl-dp').innerText=this.value" onchange="updatePlaygroundConfig('depth', parseInt(this.value))">
            </div>
        `;
    } else if (s.architecture === 'U-Net') {
        presetsHtml = `
            <option value="none">Custom</option>
            <option value="small">U-Net Small</option>
            <option value="large">U-Net Large</option>
        `;
        configHtml = `
            <div class="mb-4">
                <label class="text-sm text-muted uppercase font-semibold mb-2 block">Base Channels: <span id="pg-lbl-ch">${s.config.base_channels}</span></label>
                <input type="range" min="16" max="256" step="16" value="${s.config.base_channels}" class="slider" style="width:100%" oninput="document.getElementById('pg-lbl-ch').innerText=this.value" onchange="updatePlaygroundConfig('base_channels', parseInt(this.value))">
            </div>
            <div class="mb-4">
                <label class="text-sm text-muted uppercase font-semibold mb-2 block">Stages: <span id="pg-lbl-st">${s.config.stages}</span></label>
                <input type="range" min="1" max="5" step="1" value="${s.config.stages}" class="slider" style="width:100%" oninput="document.getElementById('pg-lbl-st').innerText=this.value" onchange="updatePlaygroundConfig('stages', parseInt(this.value))">
            </div>
        `;
    }

    controlsDiv.innerHTML = `
        <h3 class="font-heading mb-4"><i class="fa-solid fa-sliders"></i> Controls</h3>
        
        <div class="mb-4">
            <label class="text-sm text-muted uppercase font-semibold mb-2 block">Architecture</label>
            <select id="pg-arch-select" style="width:100%; padding:8px; border-radius:4px; background:var(--bg-panel); color:white; border: 1px solid var(--color-border);" onchange="changePlaygroundArch(this.value)">
                <option value="ResNet" ${s.architecture==='ResNet'?'selected':''}>ResNet</option>
                <option value="Transformer" ${s.architecture==='Transformer'?'selected':''}>Vision Transformer</option>
                <option value="U-Net" ${s.architecture==='U-Net'?'selected':''}>U-Net</option>
            </select>
        </div>

        <div class="mb-4">
            <label class="text-sm text-muted uppercase font-semibold mb-2 block">Preset</label>
            <select id="pg-preset-select" style="width:100%; padding:8px; border-radius:4px; background:var(--bg-panel); color:white; border: 1px solid var(--color-border);" onchange="applyPlaygroundPreset(this.value)">
                ${presetsHtml}
            </select>
        </div>
        
        <hr style="border-color:var(--color-border-subtle); margin: 16px 0;">
        
        ${configHtml}
        
        <hr style="border-color:var(--color-border-subtle); margin: 16px 0;">
        
        <button class="btn btn-outline" style="width:100%;" onclick="downloadPlaygroundDesign()">
            <i class="fa-solid fa-download mr-2"></i> Save Design (JSON)
        </button>
    `;
}

window.changePlaygroundArch = function(arch) {
    window.playgroundState.architecture = arch;
    if (arch === 'ResNet') window.playgroundState.config = { base_channels: 64, stages: 4, blocks_per_stage: 2 };
    else if (arch === 'Transformer') window.playgroundState.config = { hidden_size: 768, num_heads: 12, depth: 4 };
    else if (arch === 'U-Net') window.playgroundState.config = { base_channels: 64, stages: 3 };
    
    renderPlaygroundControls();
    fetchPlaygroundGraph();
}

window.applyPlaygroundPreset = function(preset) {
    if (preset === 'none') return;
    const s = window.playgroundState;
    
    if (s.architecture === 'ResNet') {
        if (preset === 'resnet18') s.config = { base_channels: 64, stages: 4, blocks_per_stage: 2 };
        if (preset === 'resnet34') s.config = { base_channels: 64, stages: 4, blocks_per_stage: 3 };
    } else if (s.architecture === 'Transformer') {
        if (preset === 'base') s.config = { hidden_size: 768, num_heads: 12, depth: 4 };
        if (preset === 'large') s.config = { hidden_size: 1024, num_heads: 16, depth: 8 };
    } else if (s.architecture === 'U-Net') {
        if (preset === 'small') s.config = { base_channels: 64, stages: 3 };
        if (preset === 'large') s.config = { base_channels: 128, stages: 4 };
    }
    
    renderPlaygroundControls();
    fetchPlaygroundGraph();
}

window.updatePlaygroundConfig = function(key, val) {
    window.playgroundState.config[key] = val;
    
    // Debounce API fetch
    clearTimeout(window.playgroundState.debounceTimer);
    window.playgroundState.debounceTimer = setTimeout(() => {
        fetchPlaygroundGraph();
    }, 300);
}

window.downloadPlaygroundDesign = function() {
    const data = JSON.stringify(window.playgroundState, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `playground_${window.playgroundState.architecture.toLowerCase()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

async function fetchPlaygroundGraph() {
    const metricsDiv = document.getElementById('pg-metrics');
    if(metricsDiv) metricsDiv.style.opacity = '0.5';

    try {
        const payload = {
            architecture: window.playgroundState.architecture,
            config: window.playgroundState.config
        };

        const res = await fetch('/api/playground/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if(!res.ok) throw new Error("API failed");
        
        const data = await res.json();
        
        // Render Center
        renderGraph('cy-playground', data.graph);
        
        // Render Right
        renderPlaygroundMetrics(data);
        
    } catch(err) {
        if(metricsDiv) {
            metricsDiv.style.opacity = '1';
            metricsDiv.innerHTML = `<p style="color:#ef4444;">Error: ${err.message}</p>`;
        }
    }
}

function renderPlaygroundMetrics(data) {
    const div = document.getElementById('pg-metrics');
    if(!div) return;
    div.style.opacity = '1';

    const m = data.metrics;
    const b = data.baseline_metrics;
    const v = data.validation;

    const renderDelta = (curr, base, inverse=false, isStr=false) => {
        if(isStr) return '';
        const diff = curr - base;
        if (diff === 0) return '';
        const isBad = inverse ? diff < 0 : diff > 0;
        const color = isBad ? '#f87171' : '#34d399';
        const sign = diff > 0 ? '+' : '';
        let valStr = diff;
        if(Math.abs(diff) >= 1e6) valStr = (diff/1e6).toFixed(1) + 'M';
        else if(Math.abs(diff) >= 1e3) valStr = (diff/1e3).toFixed(1) + 'K';
        else if(typeof diff === 'number' && !Number.isInteger(diff)) valStr = diff.toFixed(1);
        return `<span style="font-size:0.75rem; color:${color}; font-weight:bold; margin-left:8px;">${sign}${valStr}</span>`;
    };

    let paramsStr = m.params >= 1e6 ? (m.params/1e6).toFixed(1) + 'M' : (m.params/1e3).toFixed(1) + 'K';

    let validationHtml = '';
    if (!v.is_valid) {
        validationHtml = `
            <div class="card mb-4" style="background: rgba(239, 68, 68, 0.1); border-color: #ef4444;">
                <h4 style="color: #ef4444; font-size:0.9rem; margin-bottom:4px;">❌ Invalid Architecture</h4>
                <p style="color: #fca5a5; font-size:0.8rem; font-family:monospace;">${v.error}</p>
            </div>
        `;
    } else {
        if (v.anomalies.length > 0) {
            validationHtml += `
                <div class="card mb-4" style="background: rgba(245, 158, 11, 0.1); border-color: #f59e0b;">
                    <h4 style="color: #f59e0b; font-size:0.9rem; margin-bottom:4px;">⚠️ Warnings</h4>
                    <ul style="color: #fcd34d; font-size:0.8rem; padding-left:16px;">
                        ${v.anomalies.map(a => `<li>${a}</li>`).join('')}
                    </ul>
                </div>
            `;
        } else {
            validationHtml += `
                <div class="card mb-4" style="background: rgba(16, 185, 129, 0.1); border-color: #10b981;">
                    <h4 style="color: #10b981; font-size:0.9rem; margin-bottom:4px;">✅ Mathematically Valid</h4>
                    <p style="color: #6ee7b7; font-size:0.8rem;">Tensor propagation completed successfully.</p>
                </div>
            `;
        }
    }

    // Educational Insights (Rule based)
    let insightsHtml = '';
    const paramDelta = m.params - b.params;
    if (paramDelta > b.params * 1.5) {
        insightsHtml += `<div class="mb-2" style="font-size:0.85rem;"><span style="font-size:1rem; margin-right:4px;">📈</span> <strong>Massive Capacity:</strong> You have more than doubled the parameters. Watch out for overfitting.</div>`;
    } else if (paramDelta < 0) {
        insightsHtml += `<div class="mb-2" style="font-size:0.85rem;"><span style="font-size:1rem; margin-right:4px;">📉</span> <strong>Lighter Model:</strong> Parameters reduced. This helps with mobile deployment.</div>`;
    }
    
    if (window.playgroundState.architecture === "ResNet" && window.playgroundState.config.stages > 4) {
        insightsHtml += `<div class="mb-2" style="font-size:0.85rem;"><span style="font-size:1rem; margin-right:4px;">🧱</span> <strong>Deeper ResNet:</strong> Additional stages increase receptive field but gradient vanishing is prevented by skips.</div>`;
    } else if (window.playgroundState.architecture === "Transformer" && window.playgroundState.config.num_heads > 12) {
        insightsHtml += `<div class="mb-2" style="font-size:0.85rem;"><span style="font-size:1rem; margin-right:4px;">🧠</span> <strong>More Attention:</strong> Extra heads require more VRAM but allow the model to attend to more representation subspaces.</div>`;
    }

    if(!insightsHtml) insightsHtml = `<div class="text-muted text-sm">No major structural deviations from baseline.</div>`;

    div.innerHTML = `
        <h3 class="font-heading mb-4"><i class="fa-solid fa-chart-line"></i> Live Metrics</h3>
        
        ${validationHtml}

        <div class="grid gap-4" style="grid-template-columns: 1fr 1fr; margin-bottom: 16px;">
            <div class="metric-card">
                <div class="metric-label"><i class="fa-solid fa-server"></i> Parameters</div>
                <div class="metric-value" style="font-size: 1.1rem;">
                    ${paramsStr} ${renderDelta(m.params, b.params, true)}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fa-solid fa-microchip"></i> FLOPs Score</div>
                <div class="metric-value" style="font-size: 1.1rem;">
                    ${m.flops_score.toFixed(1)} ${renderDelta(m.flops_score, b.flops_score, true)}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fa-solid fa-memory"></i> Activation VRAM</div>
                <div class="metric-value" style="font-size: 1.1rem;">
                    ${m.memory_mb.toFixed(1)} MB ${renderDelta(m.memory_mb, b.memory_mb, true)}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label"><i class="fa-solid fa-layer-group"></i> Graph Depth</div>
                <div class="metric-value" style="font-size: 1.1rem;">
                    ${m.depth} ${renderDelta(m.depth, b.depth)}
                </div>
            </div>
        </div>

        <div class="card insight-card" style="background: var(--bg-panel); border: 1px solid var(--color-border-subtle);">
            <h4 class="font-heading text-sm text-muted uppercase mb-2">Educational Insights</h4>
            ${insightsHtml}
        </div>
    `;
}
