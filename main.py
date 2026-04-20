import sys
import traceback
import logging
from datetime import datetime
from pathlib import Path
import config


# ==========================================
# Logging Configuration
# ==========================================
original_stdout = sys.stdout
original_stderr = sys.stderr

# Ensure logs are saved neatly
log_filename = config.LOGS_DIR / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Bind directly to ROOT logger to capture ALL sub-modules globally
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Formatter for clean logging
formatter = logging.Formatter(
    fmt='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File Handler
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler (original stdout) so we don't cause recursion
stream_handler = logging.StreamHandler(original_stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class StreamToLogger:
    """Redirects print statements and stack traces natively to logging."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, buf):
        for line in buf.splitlines():
            # Avoid logging explicit empty lines triggered by isolated \n prints
            if line.strip():
                self.logger.log(self.level, line)

    def flush(self):
        pass

# Overtake sys.stdout and sys.stderr immediately
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

# ==========================================
# Pipeline Phases
# ==========================================
def run_phase_0():
    try:
        from runners.run_phase_0 import main as fn
    except ImportError:
        # Fallback if Phase 0 is named run_phase_0_2.py
        from runners.run_phase_0_2 import run_phase_0 as fn
    fn()

def run_phase_1a():
    from runners.run_phase_0_2 import run_phase_1a as fn
    return fn()

def run_phase_1b(df_features):
    from runners.run_phase_0_2 import run_phase_1b as fn
    return fn(df_features)

def run_phase_2(df_labeled):
    from runners.run_phase_0_2 import run_phase_2 as fn
    fn(df_labeled)

def run_phase_3():
    from runners.run_phase_3 import main as fn
    fn()

def run_phase_4():
    from runners.run_phase_4 import main as fn
    fn()

def run_phase_5():
    from runners.run_phase_5 import main as fn
    fn()

def run_phase_6():
    from runners.run_phase_6 import main as fn
    fn()

def run_phase_7():
    from runners.run_phase_7 import main as fn
    fn()

def run_phase_8():
    from runners.run_phase_8 import run_phase_8 as fn
    fn()


if __name__ == "__main__":
    logger.info("*"*60)
    logger.info(f"STARTING FULL MLR PIPELINE ALIGNMENT")
    logger.info(f"Log generated at: {log_filename}")
    logger.info("*"*60)

    try:
        # Phases 0-2 depend on returns
        logger.info(f"\n{'='*60}\n  PHASE 0: Setup & Extraction\n{'='*60}")
        run_phase_0()
        
        logger.info(f"\n{'='*60}\n  PHASE 1A: Feature Extraction\n{'='*60}")
        df_features = run_phase_1a()
        
        logger.info(f"\n{'='*60}\n  PHASE 1B: Label Generation\n{'='*60}")
        df_labeled = run_phase_1b(df_features)
        
        logger.info(f"\n{'='*60}\n  PHASE 2: Preprocessing\n{'='*60}")
        run_phase_2(df_labeled)
        
        # Isolated standalone sequences
        phases = [
            ("PHASE 3: Clustering",            run_phase_3),
            ("PHASE 4: Classification",        run_phase_4),
            ("PHASE 5: Regression (BFS)",      run_phase_5),
            ("PHASE 6: Q-Learning",            run_phase_6),
            ("PHASE 7: Adaptive Switching",    run_phase_7),
            ("PHASE 8: Final Evaluation",      run_phase_8),
        ]

        for name, fn in phases:
            logger.info(f"\n{'='*60}\n  {name}\n{'='*60}")
            fn()
            
        logger.info("\n[OK] Full pipeline executed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline halted due to fatal exception:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore sys out boundaries to prevent shell bleeding
        sys.stdout = original_stdout
        sys.stderr = original_stderr
