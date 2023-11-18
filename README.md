
# **Human-AI based Dimension Measurement in Laparoscopic Surgery**


[![Python version](https://img.shields.io/badge/python-3.10.4-green.svg)](https://www.python.org/downloads/)

![plot](/src/readme/approach.png)

A significant challenge in image-guided surgery is the accurate measurement task of relevant structures such as vessel segments, resection margins, or bowel lengths. While this task is an essential component of many surgeries, it involves substantial human effort and is prone to inaccuracies. In this paper, we develop a novel human-AI-based method for laparoscopic measurements utilizing stereo vision that has been guided by practicing surgeons. Based on a holistic qualitative requirements analysis, this work proposes a comprehensive measurement method, which comprises state-of-the-art machine learning architectures, such as RAFT-Stereo and YOLOv8. The developed method is assessed in various realistic experimental evaluation environments. Our results outline the high potential of our method achieving high accuracies in distance measurements with errors below 1 mm. Furthermore, on-surface measurements demonstrate robustness when applied in challenging environments with textureless regions. Overall, by addressing the inherent challenges of image-guided surgery, we lay the foundation for a more robust and accurate solution for intra- and postoperative measurements, enabling more precise, safe, and efficient surgical procedures.

![plot](/src/readme/exp3_point_cloud.png)

---

## Installation

Before you can start:

- Make sure you have installed a compatible **Python** version (we recommend python 3.10.4)


Now you can get your code ready:

1. Clone this repository and initialize the submodules

```
git clone https://github.com/leopoldmueller/LaparoscopicMeasurement.git
```

```
cd LaparoscopicMeasurement
```

```
git submodule init
```

```
git submodule update
```

2. Create and activate a virtual environment.

Create virtual environment:

```
# Run in terminal
python3 -m venv /path/to/new/virtual/environment
```

Activate the environment:

```
# On Unix or MacOS, using the bash shell:
source /path/to/venv/bin/activate
```

```
# On windows the cmd line:
venvironment\Scripts\activate
```


3. Install all packages

```
pip install -r requirements.txt
```

4. **Optionally** test your setup with default settings and perform sample measurements in the playground

```
python main.py
```

---

## Usage | Settings

You can set all settings in the config.yaml. So independent from your task you can always run the main.py.

## Usage | Run the measurement tool

To run the task do the following steps:

1. Activate your virtual environment.

```
# On Unix or MacOS, using the bash shell:
source /path/to/venv/bin/activate
```

```
# On windows the cmd line:
venvironment\Scripts\activate
```

2. Update your settings in the config.yaml

3. Run the main.py

```
# This will run the program
python main.py
```

## Usage | Custom data

1. To perform measurements on custom data, you need to create disparity maps first. Therefore you need to place the raft-stereo weights into the models directory. In addition you need to specify the paths in the config.yaml

2. Create the disparity maps by running the main.py

3. Now you can perform measurements using the user interface task. Set all hyperparamerters and paths in the config.yaml and run the main.py again.

4. After the image is shown on the screen you can navigate to the frame of interest using "a" and "d" keys. You can trigger the measurement using "space". Select the two reference points with the left mous button. You can zoom in and out using the right mouse button.

5. To exit the point selection mode and start the measurement press "q". The result will be printed and shown on the screen.

---

