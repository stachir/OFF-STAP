"""
scripts/19_generate_report.py

Executable script to generate the final markdown report.
"""
import sys
import os
import pandas as pd
import json

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.reporting import generate_markdown_report
from offstap.core.provenance import ProvenanceTracker
from offstap.publication import REPORT_LIMITATION


CURRENT_RUN_PRODUCTS = (
    "15_hypotheses/H1/h1_trial_pairs.csv",
    "15_hypotheses/H2/h2_summary_by_method.csv",
    "15_hypotheses/H3/h3_calibration_mode_comparison.csv",
    "baseline/baseline_summary.json",
)


def _current_run_product_appendix(base_out):
    sections = ["", "## Current-run numerical products", ""]
    for relative in CURRENT_RUN_PRODUCTS:
        path = os.path.join(base_out, *relative.split("/"))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required current-run presentation source is missing: {relative}")
        if relative.endswith(".json"):
            with open(path, encoding="utf-8") as stream:
                rendered = json.dumps(json.load(stream), indent=2)
            language = "json"
        else:
            rendered = pd.read_csv(path).to_csv(index=False).rstrip()
            language = "csv"
        sections.extend((f"### `{relative}`", "", f"```{language}", rendered, "```", ""))
    return "\n".join(sections)

def main():
    cfg = load_config("config/alignment_config.yaml")
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    base_out = cfg.paths.get("output_base", "outputs")
    metrics_path = os.path.join(base_out, "14_metrics", "evaluation_metrics.csv")
    agg_path = os.path.join(base_out, "14_metrics", "aggregated_metrics.json")

    if not os.path.exists(metrics_path) or not os.path.exists(agg_path):
        raise FileNotFoundError(f"Error: Required metric files not found in {base_out}.")

    df_metrics = pd.read_csv(metrics_path)
    with open(agg_path, 'r') as f:
        agg_metrics = json.load(f)

    report_path = os.path.join(base_out, "report.md")
    html_path = os.path.join(base_out, "report.html")

    generate_markdown_report(df_metrics, agg_metrics, report_path)

    with open(report_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    md_text = (
        f"> **Limitation:** {REPORT_LIMITATION}\n\n"
        + md_text
        + _current_run_product_appendix(base_out)
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_text)

    try:
        import markdown
        html_body = markdown.markdown(md_text, extensions=['tables'])
    except ImportError:
        # Fallback if markdown package is not installed
        html_body = f"<pre>{md_text}</pre>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pipeline Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px auto; max-width: 800px; line-height: 1.6; padding: 0 10px; color: #333; }}
            h1, h2, h3 {{ color: #111; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    prov.record_stage(
        stage_name="19_generate_report",
        config_id=cfg.config_id,
        input_files=[metrics_path, agg_path] + [os.path.join(base_out, *path.split("/")) for path in CURRENT_RUN_PRODUCTS],
        output_files=[report_path, html_path],
        additional_meta={"report_path": report_path}
    )
    print("Markdown and HTML reports generated successfully.")

if __name__ == "__main__":
    main()
