"""
core/analytics/adaptive_engine.py

Phase 9: Adaptive Learning & Personalization Engine.
Manages dynamic LearnerKnowledgeProfile, WeaknessDetection, ConceptGraph, Daily Review Plan,
and Learning Path Adaptation.
"""

from typing import Any, TypedDict


class AttemptData(TypedDict, total=False):
    question_text: str | None
    assessment_type: str | None
    architecture: str | None
    is_correct: bool


class LearnerProgressData(TypedDict, total=False):
    module_id: int
    status: str


class TutorAnalyticsData(TypedDict, total=False):
    module: str | None
    architecture: str | None
    question_count: int


class PaperModuleData(TypedDict, total=False):
    id: int
    explanation: str | None
    module_type: str | None
    layer_name: str | None


class PaperData(TypedDict, total=False):
    id: int
    title: str
    architecture_graph: dict[str, Any] | None
    modules: list[PaperModuleData]


from collections import defaultdict

CONCEPT_KEYWORDS = {
    "Convolutions": ["conv", "convolutional", "kernel", "stride", "padding", "filter"],
    "Pooling": ["pool", "pooling", "maxpool", "avgpool", "downsampling"],
    "Residual Connections": ["residual", "skip", "shortcut", "resnet", "identity"],
    "Dense Connections": ["dense", "concatenation", "densenet", "concat"],
    "Attention": ["attention", "mhsa", "self-attention", "query", "key", "value", "softmax"],
    "Transformers": [
        "transformer",
        "vit",
        "patch",
        "embedding",
        "encoderlayer",
        "multiheadattention",
    ],
    "Encoder-Decoder Architectures": [
        "unet",
        "u-net",
        "encoder-decoder",
        "decoder",
        "upsampling",
        "fcn",
        "bottleneck",
    ],
    "Tensor Shapes": ["shape", "dimension", "channel", "height", "width", "spatial", "tensor_flow"],
    "FLOPs Reasoning": ["flops", "complexity", "parameters", "estimate", "mflops", "flops_context"],
    "Architectural Tradeoffs": [
        "compare",
        "vs",
        "difference",
        "tradeoff",
        "relative",
        "architectures",
    ],
}


def _match_concept(text: str) -> str | None:
    """Helper to match text (module explanation, question, etc.) to a concept."""
    if not text:
        return None
    text_lower = text.lower()
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return concept
    return None


class AdaptiveEngine:
    """
    Computes learner profiles, weaknesses, and dynamic path adaptations.
    Fully deterministic — no LLMs used for calculation.
    """

    def compute_knowledge_profile(
        self,
        attempts: list[AttemptData],
        progress_records: list[LearnerProgressData],
        tutor_records: list[TutorAnalyticsData],
        all_modules: list[PaperModuleData],
    ) -> dict[str, float]:
        """
        Compute LearnerKnowledgeProfile (0.0 to 1.0 mastery score) for 10 core concepts.
        """
        # Load attempts, completions, and tutor queries

        # Group assessments by concept
        concept_assessments = defaultdict(list)
        for a in attempts:
            concept = _match_concept(a.get("question_text")) or _match_concept(
                a.get("assessment_type")
            )
            if not concept and a.get("architecture"):
                concept = _match_concept(a.get("architecture"))
            if concept:
                concept_assessments[concept].append(a.get("is_correct"))

        # Group completed modules by concept
        completed_mod_ids = {
            p.get("module_id") for p in progress_records if p.get("status") == "completed"
        }

        concept_total_modules = defaultdict(int)
        concept_completed_modules = defaultdict(int)
        for m in all_modules:
            concept = (
                _match_concept(m.get("explanation"))
                or _match_concept(m.get("module_type"))
                or _match_concept(m.get("layer_name"))
            )
            if concept:
                concept_total_modules[concept] += 1
                if m.get("id") in completed_mod_ids:
                    concept_completed_modules[concept] += 1

        # Group tutor queries by concept
        concept_tutor = defaultdict(int)
        for t in tutor_records:
            concept = _match_concept(t.get("module")) or _match_concept(t.get("architecture"))
            if concept:
                concept_tutor[concept] += t.get("question_count")

        profile = {}
        for concept in CONCEPT_KEYWORDS.keys():
            # 1. Assessment Accuracy
            results = concept_assessments.get(concept, [])
            accuracy = sum(1 for r in results if r) / len(results) if results else None

            # 2. Module Completion Rate
            tot_mods = concept_total_modules.get(concept, 0)
            comp_mods = concept_completed_modules.get(concept, 0)
            completion = comp_mods / tot_mods if tot_mods > 0 else 0.0

            # 3. Tutor Engagement Booster
            tutor_count = concept_tutor.get(concept, 0)
            tutor_score = min(1.0, tutor_count / 5.0)

            # Combine metrics with dynamic weights
            if accuracy is not None:
                # Learner has tested this concept
                mastery = (accuracy * 0.5) + (completion * 0.35) + (tutor_score * 0.15)
            else:
                # No tests taken: rely on module study and engagement
                mastery = (completion * 0.8) + (tutor_score * 0.2)

            profile[concept] = round(min(1.0, max(0.0, mastery)), 2)

        return profile

    def detect_weaknesses(
        self,
        attempts: list[AttemptData],
        progress_records: list[LearnerProgressData],
        tutor_records: list[TutorAnalyticsData],
        all_modules: list[PaperModuleData],
    ) -> dict[str, Any]:
        """
        Detect concepts where the learner struggles.
        Returns: { "weak_topics": [ { "topic": str, "weakness_score": float } ], "confidence": float }
        """
        # Group failures & stats by concept
        failures = defaultdict(int)
        total_tries = defaultdict(int)
        tutor_reps = defaultdict(int)
        incompletes = defaultdict(int)

        for a in attempts:
            concept = _match_concept(a.get("question_text")) or _match_concept(
                a.get("assessment_type")
            )
            if not concept and a.get("architecture"):
                concept = _match_concept(a.get("architecture"))
            if concept:
                total_tries[concept] += 1
                if not a.get("is_correct"):
                    failures[concept] += 1

        for t in tutor_records:
            concept = _match_concept(t.get("module")) or _match_concept(t.get("architecture"))
            if concept and t.get("question_count") > 1:
                tutor_reps[concept] += t.get("question_count") - 1

        completed_mod_ids = {
            p.get("module_id") for p in progress_records if p.get("status") == "completed"
        }
        for m in all_modules:
            concept = (
                _match_concept(m.get("explanation"))
                or _match_concept(m.get("module_type"))
                or _match_concept(m.get("layer_name"))
            )
            if concept and m.get("id") not in completed_mod_ids:
                incompletes[concept] += 1

        weak_topics = []
        for concept in CONCEPT_KEYWORDS.keys():
            fail_cnt = failures.get(concept, 0)
            tries = total_tries.get(concept, 0)

            # Failure rate component
            fail_rate = fail_cnt / tries if tries > 0 else 0.0

            # Tutor query repeated help search component
            rep_questions = tutor_reps.get(concept, 0)
            tutor_factor = min(1.0, rep_questions / 5.0)

            # Incompletes component
            inc_cnt = incompletes.get(concept, 0)
            inc_factor = min(1.0, inc_cnt / 10.0)

            # Calculate weakness_score
            weakness_score = (fail_rate * 0.5) + (tutor_factor * 0.25) + (inc_factor * 0.25)

            if weakness_score > 0.1:
                weak_topics.append({"topic": concept, "weakness_score": round(weakness_score, 2)})

        # Sort with highest weakness_score first
        weak_topics.sort(key=lambda x: x["weakness_score"], reverse=True)

        # Confidence: higher when we have more attempts/records
        total_data_points = len(attempts) + len(progress_records) + len(tutor_records)
        confidence = min(1.0, total_data_points / 10.0) if total_data_points > 0 else 0.1

        return {"weak_topics": weak_topics, "confidence": round(confidence, 2)}

    def get_personalized_recommendations(
        self,
        attempts: list[AttemptData],
        progress_records: list[LearnerProgressData],
        tutor_records: list[TutorAnalyticsData],
        all_modules: list[PaperModuleData],
        all_papers: list[PaperData],
    ) -> dict[str, Any]:
        """
        Dynamically compile next steps, suggested papers, modules, and tests based on weakness analysis.
        """
        weak_analysis = self.detect_weaknesses(
            attempts, progress_records, tutor_records, all_modules
        )
        profile = self.compute_knowledge_profile(
            attempts, progress_records, tutor_records, all_modules
        )

        weak_topics_list = [
            w["topic"] for w in weak_analysis["weak_topics"] if w["weakness_score"] > 0.2
        ]

        # If no weak topics, suggest whatever is lowest mastery in profile
        if not weak_topics_list:
            sorted_profile = sorted(profile.items(), key=lambda x: x[1])
            weak_topics_list = [sorted_profile[0][0]] if sorted_profile[0][1] < 0.8 else []

        review_modules = []
        suggested_assessments = []
        suggested_papers = []

        completed_mod_ids = {
            p.get("module_id") for p in progress_records if p.get("status") == "completed"
        }

        # Compile matching items based on target weak topics
        for topic in weak_topics_list:
            # 1. Suggested review modules (not completed yet)
            for paper in all_papers:
                for m in paper.get("modules", []):
                    if m.get("id") not in completed_mod_ids:
                        concept = _match_concept(m.get("explanation")) or _match_concept(
                            m.get("module_type")
                        )
                        if concept == topic:
                            review_modules.append(
                                {
                                    "module_id": m.get("id"),
                                    "paper_id": paper.get("id"),
                                    "paper_title": paper.get("title"),
                                    "layer_name": m.get("layer_name"),
                                    "topic": topic,
                                }
                            )
                            if len(review_modules) >= 3:
                                break
                if len(review_modules) >= 3:
                    break

            # 2. Suggested papers
            for paper in all_papers:
                arch_graph = paper.get("architecture_graph") or {}
                classification = arch_graph.get("classification", "")
                if (
                    _match_concept(classification) == topic
                    or _match_concept(paper.get("title")) == topic
                ):
                    suggested_papers.append(
                        {
                            "paper_id": paper.get("id"),
                            "title": paper.get("title"),
                            "classification": classification,
                            "topic": topic,
                        }
                    )
                    if len(suggested_papers) >= 2:
                        break

            # 3. Suggested assessments
            assessment_type = "tensor"
            if topic in ["FLOPs Reasoning"]:
                assessment_type = "flops"
            elif topic in ["Architectural Tradeoffs"]:
                assessment_type = "comparison"
            elif topic in ["Residual Connections", "Dense Connections", "Attention"]:
                assessment_type = "architecture"

            # Determine matching arch fallback
            target_arch = "ResNet"
            if topic in ["Dense Connections"]:
                target_arch = "DenseNet"
            elif topic in ["Encoder-Decoder Architectures"]:
                target_arch = "U-Net"
            elif topic in ["Transformers", "Attention"]:
                target_arch = "Transformer"

            suggested_assessments.append(
                {
                    "assessment_type": assessment_type,
                    "architecture": target_arch,
                    "difficulty": "beginner" if profile.get(topic, 0) < 0.4 else "intermediate",
                    "topic": topic,
                    "reason": f"Strengthen your understanding of {topic} through targeted exercises.",
                }
            )

        # Add static defaults if list is empty
        if not review_modules and all_papers:
            # Suggest first incomplete module
            for paper in all_papers:
                for m in paper.get("modules", []):
                    if m.get("id") not in completed_mod_ids:
                        review_modules.append(
                            {
                                "module_id": m.get("id"),
                                "paper_id": paper.get("id"),
                                "paper_title": paper.get("title"),
                                "layer_name": m.get("layer_name"),
                                "topic": "General",
                            }
                        )
                        break
                if review_modules:
                    break

        if not suggested_assessments:
            suggested_assessments.append(
                {
                    "assessment_type": "tensor",
                    "architecture": "ResNet",
                    "difficulty": "beginner",
                    "topic": "Tensor Shapes",
                    "reason": "Test your basic shape-tracking math.",
                }
            )

        return {
            "review_modules": review_modules[:4],
            "suggested_assessments": suggested_assessments[:3],
            "suggested_papers": suggested_papers[:3],
            "weak_topics": weak_analysis["weak_topics"][:3],
        }

    def get_daily_review_plan(
        self,
        attempts: list[AttemptData],
        progress_records: list[LearnerProgressData],
        tutor_records: list[TutorAnalyticsData],
        all_modules: list[PaperModuleData],
        all_papers: list[PaperData],
    ) -> dict[str, Any]:
        """
        Compile today's review plan featuring exactly 3 specific, targeted tasks.
        """
        recs = self.get_personalized_recommendations(
            attempts, progress_records, tutor_records, all_modules, all_papers
        )

        review_items = []

        # 1. Add module review if available
        if recs["review_modules"]:
            m = recs["review_modules"][0]
            review_items.append(
                {
                    "type": "module",
                    "title": f"Review module: {m['paper_title']} - {m['layer_name']}",
                    "url": f"#/papers/{m['paper_id']}/modules/{m['module_id']}",
                    "topic": m["topic"],
                }
            )

        # 2. Add assessment review
        if recs["suggested_assessments"]:
            a = recs["suggested_assessments"][0]
            review_items.append(
                {
                    "type": "assessment",
                    "title": f"Complete {a['difficulty'].capitalize()} {a['assessment_type'].upper()} challenge on {a['architecture']}",
                    "url": f"#/assessment?type={a['assessment_type']}&arch={a['architecture']}&difficulty={a['difficulty']}",
                    "topic": a["topic"],
                }
            )

        # 3. Add paper overview or second module
        if len(recs["review_modules"]) > 1:
            m = recs["review_modules"][1]
            review_items.append(
                {
                    "type": "module",
                    "title": f"Study block: {m['paper_title']} - {m['layer_name']}",
                    "url": f"#/papers/{m['paper_id']}/modules/{m['module_id']}",
                    "topic": m["topic"],
                }
            )
        elif recs["suggested_papers"]:
            p = recs["suggested_papers"][0]
            review_items.append(
                {
                    "type": "paper",
                    "title": f"Explore Architecture: {p['title']}",
                    "url": f"#/papers/{p['paper_id']}",
                    "topic": p["topic"],
                }
            )
        else:
            # Fallback exercise
            review_items.append(
                {
                    "type": "assessment",
                    "title": "Solve a general Tensor Shapes check",
                    "url": "#/assessment?type=tensor&arch=ResNet&difficulty=beginner",
                    "topic": "Tensor Shapes",
                }
            )

        # Ensure we have exactly 3 items
        while len(review_items) < 3:
            review_items.append(
                {
                    "type": "assessment",
                    "title": "Complete a custom FLOPs scaling verification",
                    "url": "#/assessment?type=flops&arch=ResNet&difficulty=intermediate",
                    "topic": "FLOPs Reasoning",
                }
            )

        return {"today_review": review_items[:3]}

    def get_concept_graph(
        self,
        attempts: list[AttemptData],
        progress_records: list[LearnerProgressData],
        tutor_records: list[TutorAnalyticsData],
        all_modules: list[PaperModuleData],
    ) -> list[dict[str, Any]]:
        """
        Build dynamic Concept Graph nodes with states: Mastered / Learning / Needs Review.
        """
        profile = self.compute_knowledge_profile(
            attempts, progress_records, tutor_records, all_modules
        )
        weaknesses = self.detect_weaknesses(attempts, progress_records, tutor_records, all_modules)
        weak_topics = {w["topic"]: w["weakness_score"] for w in weaknesses["weak_topics"]}

        target_concepts = [
            "CNN",
            "Residual",
            "Dense",
            "Attention",
            "Encoder Decoder",
            "FLOPs",
            "Tensor Shapes",
        ]

        # Map target graph node names to their profile keys
        profile_map = {
            "CNN": "Convolutions",
            "Residual": "Residual Connections",
            "Dense": "Dense Connections",
            "Attention": "Attention",
            "Encoder Decoder": "Encoder-Decoder Architectures",
            "FLOPs": "FLOPs Reasoning",
            "Tensor Shapes": "Tensor Shapes",
        }

        nodes = []
        for name in target_concepts:
            key = profile_map[name]
            mastery = profile.get(key, 0.0)
            w_score = weak_topics.get(key, 0.0)

            # Classify status
            if mastery >= 0.7 and w_score < 0.3:
                status = "Mastered"
            elif w_score > 0.4 or mastery < 0.3:
                status = "Needs Review"
            else:
                status = "Learning"

            nodes.append(
                {
                    "id": name.lower().replace(" ", "_"),
                    "label": name,
                    "mastery": mastery,
                    "status": status,
                }
            )

        return nodes

    def get_adaptive_learning_path(
        self,
        attempts: list[AttemptData],
        progress_records: list[LearnerProgressData],
        tutor_records: list[TutorAnalyticsData],
        all_modules: list[PaperModuleData],
    ) -> list[dict[str, Any]]:
        """
        Compile customized learning path, dynamically injecting remediation checks if needed.
        """
        profile = self.compute_knowledge_profile(
            attempts, progress_records, tutor_records, all_modules
        )

        # Baseline path structure
        path = [
            {
                "level": "Beginner",
                "focus": "Basic shapes, activations, and parameter math.",
                "papers": ["LeNet-5", "AlexNet"],
                "remediation": False,
            },
            {
                "level": "Intermediate CNNs",
                "focus": "Deep residual modeling and feature map reuse.",
                "papers": ["ResNet-18", "ResNet-50", "DenseNet-121"],
                "remediation": False,
            },
            {
                "level": "Encoder-Decoder Segmentation",
                "focus": "High-resolution spatial recovery and symmetric skips.",
                "papers": ["U-Net"],
                "remediation": False,
            },
            {
                "level": "Vision Transformers",
                "focus": "Self-attention encoders and patch embedding grids.",
                "papers": ["ViT-B/16"],
                "remediation": False,
            },
        ]

        # ── Adaptations & Remediation Injectors ──

        # 1. Prerequisite check for Residual connections
        resnet_mastery = profile.get("Residual Connections", 0.0)
        if resnet_mastery < 0.5:
            # Inject a remediation assessment check right after Beginner stage (Index 1)
            remediation_node = {
                "level": "Remediation: Residual Connections",
                "focus": "Prerequisite Check: You must master residual identity addition blocks.",
                "papers": ["Required: Residual Block Assessment"],
                "remediation": True,
                "concept": "Residual Connections",
                "mastery": resnet_mastery,
            }
            path.insert(1, remediation_node)

        # 2. Prerequisite check for Attention
        attention_mastery = profile.get("Attention", 0.0)
        # Find index of Vision Transformers stage in adapted path
        vit_idx = next(i for i, step in enumerate(path) if "Vision Transformers" in step["level"])

        if attention_mastery < 0.5:
            # Inject attention remediation before Vision Transformers
            remediation_node = {
                "level": "Remediation: Attention Mechanics",
                "focus": "Prerequisite Check: Review sequence scaling and attention projection blocks.",
                "papers": ["Required: Self-Attention Quiz"],
                "remediation": True,
                "concept": "Attention",
                "mastery": attention_mastery,
            }
            path.insert(vit_idx, remediation_node)

        return path


# Singleton instance
adaptive_engine = AdaptiveEngine()
