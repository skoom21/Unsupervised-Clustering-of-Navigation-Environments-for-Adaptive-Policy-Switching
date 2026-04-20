import os
import re

p_error = re.compile(r'print\(\s*(f?"(?:\[ERROR\]\s*)?.*?)"\s*\)')
p_warning = re.compile(r'print\(\s*(f?"(?:\[WARNING\]\s*)?.*?)"\s*\)')
p_info = re.compile(r'print\(\s*(f?"(?:\[INFO\]\s*)?.*?)"\s*\)')
p_ok = re.compile(r'print\(\s*(f?"(?:\[OK\]\s*)?.*?)"\s*\)')
p_any = re.compile(r'print\((.*)\)')

src_files = [
    "src/setup_utils.py",
    "src/data_loader.py",
    "src/label_generator.py",
    "src/preprocessing.py",
    "src/evaluation.py",
    "src/rl_agent.py",
    "src/policy_switcher.py"
]

for path in src_files:
    if os.path.exists(path):
        with open(path, "r") as f:
            content = f.read()

        if "import logging" not in content:
            if "import pandas as pd" in content:
                content = content.replace("import pandas as pd", "import logging\nimport pandas as pd\n\nlogger = logging.getLogger(__name__)")
            elif "import numpy as np" in content:
                content = content.replace("import numpy as np", "import logging\nimport numpy as np\n\nlogger = logging.getLogger(__name__)")
            else:
                parts = content.split('\n', 1)
                content = parts[0] + "\nimport logging\n\nlogger = logging.getLogger(__name__)\n" + parts[1]

        lines = content.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('print('):
                if '[ERROR]' in line:
                    line = line.replace('print(', 'logger.error(').replace('[ERROR] ', '')
                elif '[WARNING]' in line:
                    line = line.replace('print(', 'logger.warning(').replace('[WARNING] ', '')
                elif '[INFO]' in line:
                    line = line.replace('print(', 'logger.info(').replace('[INFO] ', '')
                elif '[OK]' in line:
                    line = line.replace('print(', 'logger.info(').replace('[OK] ', '')
                else:
                    line = line.replace('print(', 'logger.info(')
                lines[i] = line
                
        with open(path, "w") as f:
            f.write('\n'.join(lines))
