"""Export the notebook's bounded IK without executing its interactive example."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
usage = Path(__file__).resolve().parent
os.chdir(usage)
notebook = json.loads((usage/'robot_inverse_kinematics_analysis.ipynb').read_text())
namespace = {'__name__': '__bounded_ik_export__'}
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'IK notebook cell {i}', 'exec'), namespace)
        if 'ik_func' in namespace: break
namespace['ik_func'].save(str(output))
namespace['ik_step'].save(str(output.with_name('ik_step_bounded.casadi')))
print(f'Exported {output}')
