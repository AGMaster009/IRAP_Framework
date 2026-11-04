"""
explainer.py
------------
Explainability Module for the IRAP Framework.

Translates internal mathematical models and surrogate SHAP interactions into:
1. Feature-by-feature transparent rationale cards (Explainable Audits).
2. Knowledge Graph exports in JSON and DOT (Graphviz) formats.
3. Summary of hub features and physiological dependencies.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import networkx as nx

from src.models.dataset import Dataset


class ExplainabilityEngine:
    """
    Produces deterministic, human-readable explanations and audit trails
    for all preprocessing and relationship discovery actions taken by IRAP.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.logger = logging.getLogger("IRAP_ExplainabilityEngine")
        self.output_dir = output_dir

    def export_knowledge_graph(self, dataset: Dataset) -> Dict[str, str]:
        """
        Exports the NetworkX Context Knowledge Graph to JSON and DOT formats.
        """
        graph = dataset.relationship_graph
        if graph is None:
            return {}

        os.makedirs(self.output_dir, exist_ok=True)
        graph_json_path = os.path.join(self.output_dir, "knowledge_graph.json")
        graph_dot_path = os.path.join(self.output_dir, "knowledge_graph.dot")

        # 1. Export JSON (node-link format)
        try:
            data = nx.node_link_data(graph)
            with open(graph_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.warning(f"Could not export graph to JSON: {e}")

        # 2. Export DOT (Graphviz) format
        try:
            lines = ["digraph ContextKnowledgeGraph {", "  rankdir=LR;", '  node [shape=box, style="rounded,filled", fillcolor="#E8F0FE", fontname="Helvetica"];', '  edge [fontname="Helvetica", fontsize=10];']
            for u, v, data_edge in sorted(graph.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True):
                weight = data_edge.get("weight", 0.0)
                penwidth = max(1.0, min(5.0, weight * 4))
                lines.append(f'  "{u}" -> "{v}" [label="{weight:.2f}", penwidth={penwidth:.1f}, color="#1A73E8"];')
            lines.append("}")
            with open(graph_dot_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            self.logger.warning(f"Could not export graph to DOT: {e}")

        return {"json": graph_json_path, "dot": graph_dot_path}

    def explain(self, dataset: Dataset) -> Dataset:
        """
        Generates full explainability logs and audits for every feature.
        """
        self.logger.info("==================================================")
        self.logger.info("Starting Phase 9: Explainability & Audit Engine")
        self.logger.info("==================================================")

        explanations: List[Dict[str, Any]] = []
        graph = dataset.relationship_graph

        for plan_item in dataset.preprocessing_plan:
            feature_name = plan_item["feature"]
            profile = dataset.feature_profiles.get(feature_name)

            # Discover dependencies from Graph
            influences = []
            influenced_by = []
            if graph is not None and graph.has_node(feature_name):
                # Incoming edges: Predictors that influence this feature
                for u, _, edge_data in graph.in_edges(feature_name, data=True):
                    influences.append({
                        "predictor": u,
                        "shap_interaction_weight": round(edge_data.get("weight", 0.0), 4)
                    })
                # Outgoing edges: Features influenced by this feature
                for _, v, edge_data in graph.out_edges(feature_name, data=True):
                    influenced_by.append({
                        "target": v,
                        "shap_interaction_weight": round(edge_data.get("weight", 0.0), 4)
                    })

            # Sort dependencies by weight
            influences.sort(key=lambda x: x["shap_interaction_weight"], reverse=True)
            influenced_by.sort(key=lambda x: x["shap_interaction_weight"], reverse=True)

            # Build action rationales
            action_cards = []
            for action in plan_item.get("actions", []):
                step = action.get("step")
                method = action.get("method")
                reason = action.get("reason", "Standard protocol")
                action_cards.append({
                    "step": step,
                    "applied_method": method,
                    "mathematical_rationale": reason
                })

            feature_explanation = {
                "feature": feature_name,
                "data_type": profile.data_type if profile else "Unknown",
                "missing_percentage": round(profile.missing_percentage, 2) if profile else 0.0,
                "outlier_count": profile.outlier_count if profile else 0,
                "distribution_type": profile.distribution_type if profile else "Unknown",
                "skewness": round(profile.skewness, 3) if profile and profile.skewness is not None else None,
                "cardinality": profile.cardinality if profile else "Unknown",
                "unique_values": profile.unique_values if profile else 0,
                "discovered_dependencies": {
                    "influenced_by_predictors": influences,
                    "influences_targets": influenced_by
                },
                "decisions": action_cards,
                "guardrail_rules": profile.validation_rules if profile else []
            }

            # Update the FeatureProfile object with explanation
            if profile:
                profile.explanation = f"Decisions: {[a['applied_method'] for a in action_cards]}. Rationale: {[a['mathematical_rationale'] for a in action_cards]}"
                profile.reasoning = f"Influenced by {len(influences)} features. Influences {len(influenced_by)} features."

            explanations.append(feature_explanation)

        dataset.explanations = explanations
        paths = self.export_knowledge_graph(dataset)

        dataset.pipeline_history.append(
            f"Explainability Module completed: Generated deterministic audit cards for "
            f"{len(explanations)} features. Exported Context Knowledge Graph to {paths.get('dot', 'outputs')}."
        )
        dataset.update_modified_time()

        self.logger.info(f"Explainability Engine generated audit trails for {len(explanations)} features.")
        return dataset

