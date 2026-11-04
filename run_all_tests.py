"""
run_all_tests.py
----------------
Executes the complete system verification:
1. Unit and Integration Test Suite
2. Full Pipeline on Realistic Cohort
3. Full Pipeline on Hematology CBC Cohort
4. Full Pipeline on Hospital Diabetes Cohort (5,132 patients)
5. Multi-Dataset Benchmark and Ablation Suite
6. File and Artifact Integrity Audit

Compiles all outputs, metrics, and logs into 'outputs/full_system_test_report.txt'.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

REPORT_FILE = "outputs/full_system_test_report.txt"
os.makedirs("outputs", exist_ok=True)


def log_and_print(fh, text: str):
    print(text)
    fh.write(text + "\n")
    fh.flush()


def run_command_capture(cmd: list, title: str, fh):
    log_and_print(fh, "=" * 80)
    log_and_print(fh, f"STEP: {title}")
    log_and_print(fh, f"Command: {' '.join(cmd)}")
    log_and_print(fh, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(fh, "-" * 80)

    start = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    duration = time.time() - start

    output = proc.stdout.strip()
    if output:
        log_and_print(fh, output)

    status = "SUCCESS (Code 0)" if proc.returncode == 0 else f"FAILED (Code {proc.returncode})"
    log_and_print(fh, "-" * 80)
    log_and_print(fh, f"Result: {status} | Duration: {duration:.2f}s")
    log_and_print(fh, "=" * 80 + "\n")
    return proc.returncode == 0


def main():
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        log_and_print(fh, "********************************************************************************")
        log_and_print(fh, "        IRAP FRAMEWORK: FULL SYSTEM TEST AND VERIFICATION REPORT                ")
        log_and_print(fh, f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_and_print(fh, f"Operating System: {sys.platform} | Python: {sys.version.split()[0]}")
        log_and_print(fh, "********************************************************************************\n")

        all_passed = True

        # 1. Unit & Integration Tests
        test_ok = run_command_capture(
            [sys.executable, "-m", "unittest", "tests/test_pipeline.py"],
            "Integration and Unit Test Suite",
            fh
        )
        all_passed = all_passed and test_ok

        # 2. Pipeline on Realistic Cohort
        real_ok = run_command_capture(
            [sys.executable, "src/main.py", "--dataset", "datasets/realistic_sample.csv", "--target", "diabetes_target"],
            "IRAP Pipeline Execution: Realistic Sample Cohort (Synthetic Non-Linear)",
            fh
        )
        all_passed = all_passed and real_ok

        # 3. Pipeline on CBC Hematology Lab
        cbc_ok = run_command_capture(
            [sys.executable, "src/main.py", "--dataset", "datasets/sample/CBC data_for_meandeley_csv.csv", "--target", "Sex  "],
            "IRAP Pipeline Execution: Mendeley CBC Hematology Dataset",
            fh
        )
        all_passed = all_passed and cbc_ok

        # 4. Pipeline on Hospital Diabetes Cohort (5,132 patients)
        diab_ok = run_command_capture(
            [sys.executable, "src/main.py", "--dataset", "datasets/sample/Diabetes Classification.csv", "--target", "Diagnosis"],
            "IRAP Pipeline Execution: Hospital Diabetes Diagnostic Cohort (5,132 rows)",
            fh
        )
        all_passed = all_passed and diab_ok

        # 5. Multi-Dataset Benchmark & Ablation Suite
        bench_ok = run_command_capture(
            [sys.executable, "experiments/run_benchmarks.py"],
            "Cross-Dataset Benchmark and Ablation Experiment Suite",
            fh
        )
        all_passed = all_passed and bench_ok

        # 6. Artifact & File Integrity Audit
        log_and_print(fh, "=" * 80)
        log_and_print(fh, "STEP: Artifact and Generated Files Audit")
        log_and_print(fh, "-" * 80)

        required_artifacts = [
            ("outputs/pipeline.log", "Chronological logging file"),
            ("outputs/discovered_edges.txt", "Discovered Knowledge Graph edge list"),
            ("outputs/knowledge_graph.dot", "Graphviz DOT visualization graph"),
            ("outputs/knowledge_graph.json", "Node-link Knowledge Graph JSON"),
            ("outputs/reports/irap_execution_report.md", "Executive Markdown report"),
            ("outputs/reports/irap_execution_report.html", "Styled standalone HTML executive report"),
            ("outputs/reports/benchmark_summary.csv", "Comparative ablation summary CSV"),
            ("outputs/reports/benchmark_summary.md", "Comparative ablation summary Markdown"),
            ("datasets/biological_guardrails.json", "LOINC Biological Guardrails database"),
            ("PROJECT_MILESTONES.md", "Project Milestones Documentation"),
            ("PPT_PREPARATION.md", "Panel Review Presentation Slide Preparation"),
        ]

        missing_count = 0
        for path, desc in required_artifacts:
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            status_str = f"EXISTS ({size:,} bytes)" if exists else "MISSING"
            log_and_print(fh, f"[{status_str:^22}] {path} - {desc}")
            if not exists or size == 0:
                missing_count += 1

        log_and_print(fh, "-" * 80)
        if missing_count == 0:
            log_and_print(fh, "ARTIFACT AUDIT RESULT: ALL REQUIRED ARTIFACTS VERIFIED AND NON-EMPTY.")
        else:
            log_and_print(fh, f"ARTIFACT AUDIT WARNING: {missing_count} artifacts missing or empty!")
            all_passed = False

        # Final Verdict
        log_and_print(fh, "\n" + "=" * 80)
        if all_passed:
            log_and_print(fh, "FINAL SYSTEM VERIFICATION VERDICT: ALL TESTS, PIPELINES & BENCHMARKS PASSED (100% SUCCESS)")
        else:
            log_and_print(fh, "FINAL SYSTEM VERIFICATION VERDICT: ATTENTION REQUIRED (SOME STEPS REPORTED FAILURES)")
        log_and_print(fh, "=" * 80)


if __name__ == "__main__":
    main()

