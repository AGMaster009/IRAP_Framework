"""
report_generator.py
--------------------
Generates academic and executive audit reports for the IRAP Framework.
Outputs both Markdown (.md) and self-contained styled HTML (.html) reports.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.models.dataset import Dataset


class ReportGenerator:
    """
    Synthesizes the complete lifecycle of a Dataset into an academic,
    audit-ready evaluation report.
    """

    def __init__(self, reports_dir: str = "outputs/reports"):
        self.logger = logging.getLogger("IRAP_ReportGenerator")
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_markdown_report(self, dataset: Dataset, filename: str = "irap_execution_report.md") -> str:
        """
        Generates a comprehensive Markdown report.
        """
        filepath = os.path.join(self.reports_dir, filename)
        meta = dataset.metadata or {}
        d_profile = dataset.dataset_profile
        v_results = dataset.validation_results or {}
        graph = dataset.relationship_graph

        lines = [
            "# IRAP Framework: Automated Technical & Clinical Preprocessing Report",
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            f"- **Dataset File:** `{meta.get('filename', 'Unknown')}`",
            f"- **Raw Shape:** {dataset.get_shape()[0]} rows × {dataset.get_shape()[1]} columns",
            f"- **Processed Shape:** {dataset.processed_dataframe.shape if dataset.has_processed_data() else 'N/A'}",
            f"- **Discovered Dependencies (Knowledge Graph Edges):** {graph.number_of_edges() if graph else 0}",
            f"- **Target Variable:** `{v_results.get('target_column', 'None')}` ({v_results.get('task_type', 'Unsupervised').title()})",
            "",
            "---",
            "",
            "## 2. Dataset Health & Profiling",
        ]

        if d_profile:
            lines.extend([
                f"- **Global Missing Percentage:** {d_profile.missing_percentage:.2f}% ({d_profile.total_missing_values} cells)",
                f"- **Duplicate Rows:** {d_profile.duplicate_rows} ({d_profile.duplicate_percentage:.2f}%)",
                f"- **Feature Breakdown:** {d_profile.numeric_columns} Numeric, {d_profile.categorical_columns} Categorical, {d_profile.boolean_columns} Boolean",
                "- **Health Indicators Flagged:**",
                f"  - High Dimensionality: `{d_profile.is_high_dimensional}`",
                f"  - Heavy Missingness: `{d_profile.has_many_missing_values}`",
                f"  - Severe Outliers Present: `{d_profile.has_outliers}`",
                f"  - High Cardinality Present: `{d_profile.has_high_cardinality}`",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## 3. Discovered Knowledge Graph & Dependencies",
            "Top non-linear feature relationships discovered via TreeSHAP interaction extraction:",
            "",
            "| Predictor Feature | Target Feature | SHAP Interaction Weight | Clinical / Mathematical Meaning |",
            "| :--- | :--- | :---: | :--- |",
        ])

        if graph and graph.number_of_edges() > 0:
            top_edges = sorted(graph.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)[:10]
            for u, v, data in top_edges:
                w = data.get("weight", 0.0)
                meaning = "Direct functional dependence" if w > 0.5 else "Moderate interaction"
                lines.append(f"| **{u}** | **{v}** | `{w:.4f}` | {meaning} |")
        else:
            lines.append("| *No strong dependencies discovered* | - | - | - |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Intelligent Preprocessing Plan & Applied Actions",
            "Deterministic preprocessing actions decided by the IRAP Engine:",
            "",
            "| Feature | Missing % | Applied Imputation | Applied Encoding | Applied Scaling | Rationale |",
            "| :--- | :---: | :--- | :--- | :--- | :--- |",
        ])

        for plan_item in dataset.preprocessing_plan:
            feat = plan_item["feature"]
            prof = dataset.feature_profiles.get(feat)
            missing_pct = f"{prof.missing_percentage:.1f}%" if prof else "0.0%"

            imp_act = next((a for a in plan_item.get("actions", []) if a.get("step") == "imputation"), None)
            enc_act = next((a for a in plan_item.get("actions", []) if a.get("step") == "encoding"), None)
            sca_act = next((a for a in plan_item.get("actions", []) if a.get("step") == "scaling"), None)

            imp_str = f"`{imp_act['method']}`" if imp_act else "None"
            enc_str = f"`{enc_act['method']}`" if enc_act else "None"
            sca_str = f"`{sca_act['method']}`" if sca_act else "None"

            reasons = []
            if imp_act:
                reasons.append(imp_act.get("reason", ""))
            if enc_act:
                reasons.append(enc_act.get("reason", ""))
            if sca_act:
                reasons.append(sca_act.get("reason", ""))
            rationale_str = "; ".join(reasons) if reasons else "No transformation needed"

            lines.append(f"| **{feat}** | {missing_pct} | {imp_str} | {enc_str} | {sca_str} | {rationale_str} |")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Validation & Comparative Benchmarking (Ablation Study)",
            "Downstream model performance comparing standard Scikit-Learn baseline vs IRAP:",
            "",
        ])

        if v_results:
            base_m = v_results.get("baseline_metrics", {})
            irap_m = v_results.get("irap_metrics", {})
            imp_m = v_results.get("improvements", {})
            task = v_results.get("task_type", "classification")

            if task == "classification":
                lines.extend([
                    "| Evaluation Metric | Baseline (Scikit-Learn Standard) | IRAP Framework | Relative Improvement |",
                    "| :--- | :---: | :---: | :---: |",
                    f"| **Accuracy** | {base_m.get('accuracy', 'N/A')} | **{irap_m.get('accuracy', 'N/A')}** | `{imp_m.get('delta_accuracy_percent', 0.0):+}%` |",
                    f"| **F1-Score (Weighted)** | {base_m.get('f1_score', 'N/A')} | **{irap_m.get('f1_score', 'N/A')}** | `{imp_m.get('delta_f1_percent', 0.0):+}%` |",
                    f"| **Precision** | {base_m.get('precision', 'N/A')} | **{irap_m.get('precision', 'N/A')}** | - |",
                    f"| **Recall** | {base_m.get('recall', 'N/A')} | **{irap_m.get('recall', 'N/A')}** | - |",
                ])
            else:
                lines.extend([
                    "| Evaluation Metric | Baseline (Scikit-Learn Standard) | IRAP Framework | Difference |",
                    "| :--- | :---: | :---: | :---: |",
                    f"| **R² Score** | {base_m.get('r2_score', 'N/A')} | **{irap_m.get('r2_score', 'N/A')}** | `{imp_m.get('delta_r2', 0.0):+}` |",
                    f"| **RMSE** | {base_m.get('rmse', 'N/A')} | **{irap_m.get('rmse', 'N/A')}** | `{-imp_m.get('mse_reduction', 0.0):+}` |",
                ])

            fidelity = v_results.get("imputation_fidelity", {})
            if fidelity:
                lines.extend([
                    "",
                    "### Imputation Fidelity (Ground Truth Reconstruction):",
                    f"- **Tested Feature:** `{fidelity.get('evaluated_feature')}`",
                    f"- **Baseline Imputation RMSE:** `{fidelity.get('baseline_imputation_rmse')}`",
                    f"- **IRAP Context-Aware Imputation RMSE:** `{fidelity.get('irap_imputation_rmse')}`",
                    f"- **Error Reduction:** **`{fidelity.get('rmse_reduction_percent')}%` reduction in imputation error**",
                ])

        lines.extend([
            "",
            "---",
            "",
            "## 6. Execution Audit Trail",
            "Full chronological log of pipeline execution:",
            "",
        ])

        for step in dataset.pipeline_history:
            lines.append(f"- {step}")

        content = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        self.logger.info(f"Saved Markdown report to: {filepath}")
        return filepath

    def generate_html_report(self, dataset: Dataset, filename: str = "irap_execution_report.html") -> str:
        """
        Generates a standalone styled HTML report for presentations and defense documentation.
        """
        filepath = os.path.join(self.reports_dir, filename)
        meta = dataset.metadata or {}
        v_results = dataset.validation_results or {}
        base_m = v_results.get("baseline_metrics", {})
        irap_m = v_results.get("irap_metrics", {})
        imp_m = v_results.get("improvements", {})
        fidelity = v_results.get("imputation_fidelity", {})
        graph = dataset.relationship_graph

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IRAP Framework - Execution & Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; padding: 24px; background: #f8f9fa; color: #202124; line-height: 1.5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
        h1 {{ color: #1a73e8; margin-top: 0; border-bottom: 2px solid #e8eaed; padding-bottom: 12px; font-size: 26px; }}
        h2 {{ color: #202124; margin-top: 28px; font-size: 20px; border-left: 4px solid #1a73e8; padding-left: 10px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 13px; font-weight: bold; }}
        .badge-success {{ background: #e6f4ea; color: #137333; }}
        .badge-info {{ background: #e8f0fe; color: #1a73e8; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
        .card {{ background: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 16px; text-align: center; }}
        .card .metric {{ font-size: 28px; font-weight: bold; color: #1a73e8; margin-top: 8px; }}
        .card .label {{ font-size: 13px; color: #5f6368; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ padding: 10px 12px; border: 1px solid #dadce0; text-align: left; font-size: 14px; }}
        th {{ background: #f1f3f4; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .highlight {{ font-weight: bold; color: #137333; }}
        .audit-list {{ background: #202124; color: #e8eaed; padding: 16px 20px; border-radius: 8px; font-family: monospace; font-size: 13px; list-style: none; }}
        .audit-list li {{ margin-bottom: 6px; }}
        .audit-list li::before {{ content: "✓ "; color: #34a853; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>IRAP Framework: Technical & Clinical Validation Report</h1>
        <p><strong>Dataset:</strong> {meta.get('filename', 'Unknown')} | <strong>Execution Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="card-grid">
            <div class="card">
                <div class="label">Total Patients / Rows</div>
                <div class="metric">{dataset.get_shape()[0]}</div>
            </div>
            <div class="card">
                <div class="label">Knowledge Graph Edges</div>
                <div class="metric">{graph.number_of_edges() if graph else 0}</div>
            </div>
            <div class="card">
                <div class="label">Downstream Accuracy</div>
                <div class="metric">{irap_m.get('accuracy', 'N/A')}</div>
            </div>
            <div class="card">
                <div class="label">Imputation Error Reduction</div>
                <div class="metric">{fidelity.get('rmse_reduction_percent', 0.0)}%</div>
            </div>
        </div>

        <h2>Comparative Validation (IRAP vs Scikit-Learn Baseline)</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Standard Scikit-Learn Baseline</th>
                    <th>IRAP Framework</th>
                    <th>Gain / Delta</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Model Accuracy</strong></td>
                    <td>{base_m.get('accuracy', 'N/A')}</td>
                    <td><strong>{irap_m.get('accuracy', 'N/A')}</strong></td>
                    <td class="highlight">{imp_m.get('delta_accuracy_percent', 0.0):+}%</td>
                </tr>
                <tr>
                    <td><strong>F1-Score (Weighted)</strong></td>
                    <td>{base_m.get('f1_score', 'N/A')}</td>
                    <td><strong>{irap_m.get('f1_score', 'N/A')}</strong></td>
                    <td class="highlight">{imp_m.get('delta_f1_percent', 0.0):+}%</td>
                </tr>
                <tr>
                    <td><strong>Imputation RMSE (Feature: {fidelity.get('evaluated_feature', 'N/A')})</strong></td>
                    <td>{fidelity.get('baseline_imputation_rmse', 'N/A')}</td>
                    <td><strong>{fidelity.get('irap_imputation_rmse', 'N/A')}</strong></td>
                    <td class="highlight">-{fidelity.get('rmse_reduction_percent', 0.0)}% Error</td>
                </tr>
            </tbody>
        </table>

        <h2>Chronological Pipeline Audit Trail</h2>
        <ul class="audit-list">
            {"".join(f"<li>{step}</li>" for step in dataset.pipeline_history)}
        </ul>
    </div>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"Saved HTML report to: {filepath}")
        return filepath

    def generate(self, dataset: Dataset) -> Dict[str, str]:
        """
        Generates both Markdown and HTML reports.
        """
        self.logger.info("Generating final evaluation reports...")
        md_path = self.generate_markdown_report(dataset)
        html_path = self.generate_html_report(dataset)
        return {"markdown": md_path, "html": html_path}

