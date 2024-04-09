# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

import os


######## Basic ########

# Folder with the BOP datasets.
if "BOP_PATH" in os.environ:
    datasets_path = os.environ["BOP_PATH"]
else:
    datasets_path = r"..\\dataset" 

# Folder with pose results to be evaluated.
results_path = r"..\\result\\YOLO-ICP\\"   # r"..\\result\\hodan-iros15\\"  "..\\result\\YOLO-ICP\\"

# Folder for the calculated pose errors and performance scores.
eval_path = r"..\\result\\YOLO-ICP\\"

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r"..\\result\\YOLO-ICP\\"

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r"..\\result"

# Executable of the MeshLab server.
meshlab_server_path = r"/path/to/meshlabserver.exe"
