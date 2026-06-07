import re

with open("static/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Inject CSS
css_injection = """
        /* Tutor Panel */
        .tutor-panel { position: fixed; top: 0; right: -400px; width: 400px; height: 100vh; background: var(--bg-surface); border-left: 1px solid var(--color-border); box-shadow: -4px 0 24px rgba(0,0,0,0.2); z-index: 1000; display: flex; flex-direction: column; transition: right 0.3s ease; }
        .tutor-panel.open { right: 0; }
        .tutor-header { padding: var(--space-4); border-bottom: 1px solid var(--color-border); display: flex; justify-content: space-between; align-items: center; }
        .tutor-chat { flex: 1; overflow-y: auto; padding: var(--space-4); display: flex; flex-direction: column; gap: var(--space-3); }
        .tutor-msg { padding: var(--space-3); border-radius: 8px; font-size: 0.9rem; line-height: 1.5; max-width: 90%; }
        .tutor-msg.user { background: var(--bg-panel); align-self: flex-end; }
        .tutor-msg.bot { background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2); align-self: flex-start; }
        .tutor-msg-meta { font-size: 0.7rem; color: var(--color-text-secondary); margin-top: 4px; border-top: 1px solid var(--color-border); padding-top: 4px; }
        .tutor-input-area { padding: var(--space-4); border-top: 1px solid var(--color-border); display: flex; gap: var(--space-2); }
        .tutor-input-area input { flex: 1; padding: var(--space-2); border-radius: 6px; border: 1px solid var(--color-border); background: var(--bg-body); color: white; }
"""
html = html.replace("/* Mobile Adjustments */", css_injection + "\n        /* Mobile Adjustments */")

# 2. Inject HTML for Panel
panel_html = """
    <!-- Tutor Panel -->
    <div id="tutor-panel" class="tutor-panel">
        <div class="tutor-header">
            <h3><i class="fa-solid fa-robot" style="color: var(--accent-transformer)"></i> Architecture Tutor</h3>
            <button onclick="closeTutor()" class="btn btn-secondary btn-sm"><i class="fa-solid fa-times"></i></button>
        </div>
        <div id="tutor-chat" class="tutor-chat"></div>
        <div class="tutor-input-area">
            <input type="text" id="tutor-input" placeholder="Ask about this context..." onkeypress="if(event.key === 'Enter') sendTutorMessage()">
            <button onclick="sendTutorMessage()" class="btn btn-primary"><i class="fa-solid fa-paper-plane"></i></button>
        </div>
    </div>
"""
html = html.replace('<div id="cy-tooltip" class="cy-tooltip"></div>', '<div id="cy-tooltip" class="cy-tooltip"></div>\n' + panel_html)

# 3. Inject JS Logic
js_injection = """
        // Tutor Logic
        let tutorSessionId = "session_" + Math.random().toString(36).substr(2, 9);
        let tutorCurrentContext = {};
        let tutorContextType = "none";

        function openTutor(contextType, contextData) {
            tutorContextType = contextType;
            tutorCurrentContext = contextData;
            document.getElementById("tutor-panel").classList.add("open");
            const chat = document.getElementById("tutor-chat");
            if(chat.children.length === 0) {
                appendTutorMessage("bot", "Hello! I am your AI Architecture Tutor. I have context on the current " + contextType + ". What would you like to know?");
            }
        }
        function closeTutor() { document.getElementById("tutor-panel").classList.remove("open"); }
        function appendTutorMessage(role, text, meta = null) {
            const chat = document.getElementById("tutor-chat");
            const div = document.createElement("div");
            div.className = `tutor-msg ${role}`;
            let htmlText = text.replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
            div.innerHTML = htmlText;
            if (meta) {
                div.innerHTML += `<div class="tutor-msg-meta">
                    <div><i class="fa-solid fa-bullseye"></i> ${meta.source_context}</div>
                    <div><i class="fa-solid fa-brain"></i> ${meta.reasoning_type} (${meta.confidence})</div>
                </div>`;
            }
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function sendTutorMessage() {
            const input = document.getElementById("tutor-input");
            const text = input.value.trim();
            if(!text) return;
            input.value = "";
            appendTutorMessage("user", text);
            const typingId = "typing_" + Date.now();
            const typingDiv = document.createElement("div");
            typingDiv.id = typingId; typingDiv.className = "tutor-msg bot text-muted";
            typingDiv.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Thinking...';
            document.getElementById("tutor-chat").appendChild(typingDiv);
            document.getElementById("tutor-chat").scrollTop = document.getElementById("tutor-chat").scrollHeight;

            try {
                const res = await fetch("/api/tutor/ask", {
                    method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ session_id: tutorSessionId, context_type: tutorContextType, context_data: tutorCurrentContext, query: text })
                });
                const data = await res.json();
                document.getElementById(typingId).remove();
                appendTutorMessage("bot", data.answer, data);
            } catch(e) {
                document.getElementById(typingId).remove();
                appendTutorMessage("bot", "Error connecting to tutor.");
            }
        }

        async function loadLearningPath() {
            appContainer.innerHTML = `<div class="text-center mt-8"><div class="spinner"></div><p>Generating Learning Path...</p></div>`;
            navActions.innerHTML = '';
            
            try {
                const res = await fetch("/api/tutor/learning-path");
                const data = await res.json();
                
                let pathHtml = `<div class="page-view"><h1 class="hero-title mb-4">Architecture Learning Path</h1><p class="text-muted mb-8">A deterministic sequence for learning deep learning architectures.</p><div style="display: flex; flex-direction: column; gap: var(--space-4);">`;
                
                data.learning_path.forEach((level, index) => {
                    pathHtml += `
                        <div class="metric-card" style="border-left: 4px solid var(--accent-transformer)">
                            <h3>Step ${index + 1}: ${level.level}</h3>
                            <p class="text-muted mb-4">${level.focus}</p>
                            <div style="display: flex; gap: var(--space-2); flex-wrap: wrap;">
                                ${level.papers.map(p => `<span class="badge" style="background: var(--bg-panel); padding: 4px 12px; border-radius: 16px;">${p}</span>`).join("")}
                            </div>
                        </div>
                    `;
                });
                
                pathHtml += `</div></div>`;
                appContainer.innerHTML = pathHtml;
            } catch(e) {
                appContainer.innerHTML = `<div class="text-danger">Failed to load learning path: ${e.message}</div>`;
            }
        }
"""
html = html.replace('const appContainer = document.getElementById(\'app-container\');', js_injection + '\n        const appContainer = document.getElementById(\'app-container\');')

# 4. Inject buttons into views
# Overview
overview_target = '        appContainer.innerHTML = `'
overview_button = """
        let tutorData = { title: paper.title, abstract: paper.abstract, module_count: modules.length, total_flops_score: paper.flops_analysis?.total_flops_score };
        appContainer.innerHTML = `
            <div class="flex justify-between items-center mb-4">
                <h1 class="hero-title">${paper.title}</h1>
                <button onclick='openTutor("architecture", ${JSON.stringify(tutorData).replace(/'/g, "\\'")})' class="btn btn-primary"><i class="fa-solid fa-robot"></i> Ask About This Architecture</button>
            </div>
"""
html = html.replace('appContainer.innerHTML = `\n            <h1 class="hero-title mb-2">${paper.title}</h1>', overview_button)

# Module view
module_target = '            <div class="flex justify-between items-center mb-4">\n                <h1 class="hero-title">${mod.layer_name}</h1>'
module_button = """            <div class="flex justify-between items-center mb-4">
                <h1 class="hero-title">${mod.layer_name}</h1>
                <button onclick='openTutor("module", {paper_title: "${paper.title}", layer_name: "${mod.layer_name}", module_type: "${mod.module_type}", explanation: ${JSON.stringify(mod.explanation)}, flops_context: ${JSON.stringify(mod.flops_context)}})' class="btn btn-primary"><i class="fa-solid fa-robot"></i> Ask About This Module</button>
"""
html = html.replace(module_target, module_button)

# Inject Quiz into Module
quiz_injection = """
                <div class="mt-8 mb-4">
                    <h3><i class="fa-solid fa-clipboard-question"></i> Learning Questions</h3>
                    <div id="quiz-container" class="mt-4">
                        <button onclick="generateQuiz()" class="btn btn-secondary"><i class="fa-solid fa-magic"></i> Generate Quiz for this Module</button>
                    </div>
                </div>
            </div>
"""
html = html.replace('</div>\n            </div>\n        `;\n        \n        // Render Graph', quiz_injection + '        `;\n        \n        // Render Graph')

quiz_js = """
        window.generateQuiz = async function() {
            const qc = document.getElementById("quiz-container");
            qc.innerHTML = '<div class="spinner"></div>';
            try {
                const res = await fetch("/api/tutor/quiz", {
                    method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ module_data: { layer_name: currentModuleData.layer_name, module_type: currentModuleData.module_type, explanation: currentModuleData.explanation, paper_title: currentModuleData.paper_title } })
                });
                const data = await res.json();
                let qHtml = '<div style="display: flex; flex-direction: column; gap: var(--space-4);">';
                data.questions.forEach((q, i) => {
                    qHtml += `<div class="metric-card">
                        <div class="font-semibold mb-2">Q${i+1}: ${q.question}</div>
                        <details><summary class="text-muted" style="cursor: pointer;">Show Answer</summary><div class="mt-2 text-sm">${q.answer}</div></details>
                    </div>`;
                });
                qHtml += '</div>';
                qc.innerHTML = qHtml;
            } catch(e) {
                qc.innerHTML = `<div class="text-danger">Failed to load quiz.</div>`;
            }
        };
"""
html = html.replace('const navActions = document.getElementById(\'nav-actions\');', 'const navActions = document.getElementById(\'nav-actions\');\n' + quiz_js + '\nlet currentModuleData = {};\n')

module_data_save = """
        currentModuleData = { layer_name: mod.layer_name, module_type: mod.module_type, explanation: mod.explanation, paper_title: paper.title };
"""
html = html.replace('const nodes = mod.graph_nodes || [];', module_data_save + '        const nodes = mod.graph_nodes || [];')

# Inject Learning Path Nav
nav_injection = """
        <a href="#/learning-path" class="sidebar-link"><i class="fa-solid fa-graduation-cap"></i> Learning Path</a>
"""
html = html.replace('<a href="#/playground" class="sidebar-link"><i class="fa-solid fa-flask"></i> Playground</a>', '<a href="#/playground" class="sidebar-link"><i class="fa-solid fa-flask"></i> Playground</a>' + nav_injection)

# Global routing for Learning path
router_injection = """
            } else if (hash === '#/learning-path') {
                loadLearningPath();
"""
html = html.replace('} else if (hash.startsWith(\'#/playground\')) {', router_injection + '            } else if (hash.startsWith(\'#/playground\')) {')

with open("static/index.html", "w", encoding="utf-8") as f:
    f.write(html)
