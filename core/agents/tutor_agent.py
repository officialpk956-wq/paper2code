import json
import time
from typing import Dict, Any, List
from core.llm_client import llm_complete

class TutorSessionManager:
    def __init__(self):
        # session_id -> {"last_active": timestamp, "history": [...]}
        self.sessions = {}
        self.SESSION_TIMEOUT = 30 * 60  # 30 minutes

    def _cleanup(self):
        now = time.time()
        expired = [sid for sid, data in self.sessions.items() if now - data["last_active"] > self.SESSION_TIMEOUT]
        for sid in expired:
            del self.sessions[sid]

    def _get_history(self, session_id: str) -> List[Dict[str, str]]:
        self._cleanup()
        if session_id not in self.sessions:
            self.sessions[session_id] = {"last_active": time.time(), "history": []}
        else:
            self.sessions[session_id]["last_active"] = time.time()
        return self.sessions[session_id]["history"]

    def _add_history(self, session_id: str, role: str, content: str):
        history = self._get_history(session_id)
        history.append({"role": role, "content": content})
        # Keep last 6 messages to avoid token bloat
        if len(history) > 6:
            self.sessions[session_id]["history"] = history[-6:]

    def _build_context_summary(self, context_type: str, context_data: Dict[str, Any]) -> str:
        # Constraint: Do not send full architecture graphs to the LLM.
        # Build context-specific summaries.
        if context_type == "module":
            return f"Paper/Arch: {context_data.get('paper_title')}\nModule: {context_data.get('layer_name')}\nType: {context_data.get('module_type')}\nExplanation: {context_data.get('explanation')}\nFLOPs Context: {json.dumps(context_data.get('flops_context', {}))}"
        elif context_type == "architecture":
            # Just send basic metadata, params, flops summary, etc.
            return f"Architecture: {context_data.get('title')}\nAbstract: {context_data.get('abstract')}\nTotal Modules: {context_data.get('module_count')}"
        elif context_type == "implementation":
            # Phase 10: Implementation context — include module type, PyTorch component, and design rationale
            impl = context_data.get("implementation", {})
            return (
                f"Paper/Arch: {context_data.get('paper_title')}\n"
                f"Module: {context_data.get('layer_name')}\n"
                f"Module Type: {context_data.get('module_type')}\n"
                f"PyTorch Component: {impl.get('component', 'Unknown')}\n"
                f"Concept: {impl.get('concept', '')}\n"
                f"Design Rationale: {impl.get('design_rationale', '')}\n"
                f"Implementation Label: {impl.get('label', 'Educational Implementation')}"
            )
        elif context_type == "lab":
            # Phase 11: Research Lab context — mutation effects, architecture diff, tradeoff context
            diff = context_data.get("diff", {})
            mutations = context_data.get("mutations_applied", [])
            mut_names = ", ".join(m.get("type", "?") for m in mutations) if mutations else "none"
            param_d = diff.get("param_delta", {})
            flops_d = diff.get("flops_delta", {})
            return (
                f"Architecture: {context_data.get('architecture', 'Unknown')}\n"
                f"Mutations Applied: {mut_names}\n"
                f"Structural Change: {diff.get('summary_text', 'No diff data')}\n"
                f"Parameter Delta: {param_d.get('absolute', 0):+d} ({param_d.get('pct', 0):+.1f}%)\n"
                f"FLOPs Delta: {flops_d.get('absolute', 0):+d}\n"
                f"Depth Delta: {diff.get('depth_delta', {}).get('absolute', 0):+d}\n"
                f"Skip Connections Delta: {diff.get('skip_delta', {}).get('absolute', 0):+d}\n"
                f"Memory Delta: {diff.get('memory_delta', {}).get('absolute_mb', 0):+.2f} MB\n"
                f"User Hypothesis: {context_data.get('user_hypothesis', 'Not provided')}"
            )
        else:
            return str(context_data)[:500]

    def ask(self, session_id: str, context_type: str, context_data: Dict[str, Any], query: str, knowledge_profile: Dict[str, float] = None) -> Dict[str, Any]:
        if not context_data:
            return {
                "answer": "I don't have enough architecture context.",
                "source_context": "None",
                "confidence": "Low",
                "reasoning_type": "Fallback"
            }
        
        context_summary = self._build_context_summary(context_type, context_data)
        history = self._get_history(session_id)
        
        # Format history for prompt
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
        
        profile_instructions = ""
        if knowledge_profile:
            mastered = [concept for concept, score in knowledge_profile.items() if score >= 0.7]
            weak = [concept for concept, score in knowledge_profile.items() if score < 0.3]
            
            if mastered:
                profile_instructions += f"\n- The learner has already mastered: {', '.join(mastered)}. Keep explanations relating to these concepts extremely concise, direct, and skip the basic introductory concepts."
            if weak:
                profile_instructions += f"\n- The learner struggles with: {', '.join(weak)}. Explain these concepts with detailed step-by-step math, code annotations, and structural context."

        prompt = f"""
You are the Paper2Code AI Architecture Tutor.
Answer the user's question about the provided architecture context.
Identify the architecture or module by name in your answer.
{profile_instructions}

You MUST respond with valid JSON matching exactly this schema:
{{
    "answer": "Your detailed explanation here, mentioning the specific architecture or module used.",
    "source_context": "The specific parts of the context you used to answer.",
    "confidence": "High/Medium/Low",
    "reasoning_type": "Structural/Mathematical/Intuitive"
}}

Context Type: {context_type}
Context Data:
{context_summary}

Chat History:
{history_text}

User Query: {query}
"""
        # Call LLM
        try:
            response_text = llm_complete(prompt)
            # Find JSON block if wrapped
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            parsed = json.loads(response_text)
            
            # Check for fallback
            if "i don't have enough" in parsed.get("answer", "").lower():
                parsed["answer"] = "I don't have enough architecture context."
            
            # Save to history
            self._add_history(session_id, "user", query)
            self._add_history(session_id, "tutor", parsed["answer"])
            
            return parsed
        except Exception as e:
            # Fallback deterministic mechanism if LLM fails
            return {
                "answer": f"I don't have enough architecture context. Ensure Groq API is reachable. (Error: {str(e)})",
                "source_context": "Error Fallback",
                "confidence": "Low",
                "reasoning_type": "Fallback"
            }

    def generate_quiz(self, module_data: Dict[str, Any], weak_topics: List[str] = None) -> List[Dict[str, Any]]:
        # Requirement: Quiz generation must be metadata-driven (deterministic fallback).
        questions = []
        layer_name = module_data.get('layer_name', 'This module')
        mod_type = module_data.get('module_type', '').lower()
        explanation = module_data.get('explanation', '')
        arch_name = module_data.get('paper_title', 'this architecture')
        
        # Phase 9: Prioritize Weak Topics if present
        if weak_topics:
            for topic in weak_topics:
                if topic == "FLOPs Reasoning":
                    questions.append({
                        "question": f"Given {layer_name} within {arch_name}, how does its FLOP complexity scale if you double the input channel count?",
                        "answer": "FLOPs scale quadratically with channels for layers with channel-to-channel transitions (like Conv layers) since operations ∝ C_in * C_out."
                    })
                elif topic == "Tensor Shapes":
                    questions.append({
                        "question": f"What is the input and output tensor shape mismatch risk for {layer_name} under varying spatial resolutions?",
                        "answer": "Spatial shape changes (via stride/padding or pooling) can cause addition mismatches at convergence points (like residuals) unless projected by a matching Conv."
                    })
                elif topic == "Residual Connections" and "residual" not in mod_type:
                    questions.append({
                        "question": f"How would adding a skip/residual connection around {layer_name} affect its backpropagation dynamics?",
                        "answer": "A skip connection adds an identity mapping path, allowing the gradient to flow directly backwards without scaling down, preventing vanishing gradients."
                    })
                elif topic == "Attention" and "attention" not in mod_type:
                    questions.append({
                        "question": f"How could sequence-level global dependencies be modeled in {layer_name} using Attention?",
                        "answer": "By computing self-attention query-key weights, letting each spatial location attend to all others, instead of local kernel filtering."
                    })

        if "conv" in mod_type:
            questions.append({
                "question": f"Why is a convolutional layer used in {layer_name} within {arch_name}?",
                "answer": "Convolutional layers extract spatial features like edges and textures using shared weights, preserving spatial hierarchies."
            })
        elif "pool" in mod_type:
            questions.append({
                "question": f"What is the primary effect of the pooling operation in {layer_name}?",
                "answer": "It reduces spatial dimensions, effectively downsampling the feature map to increase translation invariance and reduce computational load."
            })
        elif "attention" in mod_type or "mhsa" in mod_type:
            questions.append({
                "question": f"How does the attention mechanism in {layer_name} differ from standard convolution?",
                "answer": "Attention computes a global receptive field by weighing the importance of all tokens across the sequence, whereas convolution uses a fixed local receptive field."
            })
        elif "residual" in mod_type or "bottleneck" in mod_type or "skip" in explanation.lower():
            questions.append({
                "question": f"What is the architectural purpose of the skip connection in {layer_name}?",
                "answer": "Skip connections allow gradients to flow directly through the network, mitigating the vanishing gradient problem in deep architectures like ResNet."
            })
            
        # Add generic structural question based on explanation
        questions.append({
            "question": f"Based on its design, what is the main structural role of {layer_name}?",
            "answer": explanation
        })
        
        return questions[:5]

    def get_learning_path(self) -> List[Dict[str, Any]]:
        # Requirement: Learning Path must be deterministic.
        return [
            {
                "level": "Beginner",
                "papers": ["LeNet-5", "AlexNet", "VGG16", "VGG19"],
                "focus": "Fundamentals of spatial feature extraction and hierarchical deep learning."
            },
            {
                "level": "Intermediate",
                "papers": ["ResNet18", "ResNet34", "ResNet50", "DenseNet121", "FCN", "U-Net"],
                "focus": "Advanced connectivity, skip connections, dense feature reuse, and encoder-decoder design."
            },
            {
                "level": "Advanced",
                "papers": ["MobileNetV2", "EfficientNet-B0", "Transformer", "Vision Transformer"],
                "focus": "Compute optimization, depthwise separable convolutions, and attention mechanisms."
            }
        ]

tutor_manager = TutorSessionManager()
