import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import config
import logging

logger = logging.getLogger(__name__)
from src.evaluation import compile_master_metrics, generate_comparison_tables

def run_phase_8() -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 8: Evaluation and Comparative Analysis")
    logger.info("=" * 60)

    reports_dir = config.OUTPUTS_DIR / "reports"

    # 1. Compile Master Metrics
    logger.info("Compiling master metrics...")
    master_wide, master_long = compile_master_metrics(reports_dir)
    
    wide_path = reports_dir / "master_metrics_wide.csv"
    long_path = reports_dir / "master_metrics.csv"
    
    master_wide.to_csv(wide_path, index=False)
    master_long.to_csv(long_path, index=False)
    
    logger.info(f"Saved wide format metrics to {wide_path}")
    logger.info(f"Saved long format metrics to {long_path}")

    # 2. Generate Comparison Tables
    logger.info("\nGenerating comparison tables...\n")
    tables = generate_comparison_tables(reports_dir)
    for title, md_str in tables.items():
        logger.info(f"=== {title} ===")
        logger.info(md_str)
        logger.info("\n")
        
    logger.info("Phase 8 Final Evaluation Complete")

if __name__ == "__main__":
    run_phase_8()
