
# **Human-AI based Dimension Measurement in Laparoscopic Surgery**


[![Python version](https://img.shields.io/badge/python-3.10.4-green.svg)](https://www.python.org/downloads/)

![plot](/src/readme/approach.png)

Laparoscopic surgery poses unique challenges for accurate measurement of structures due to limited working space, specialized instruments, and distorted camera perspectives. In response to these challenges, we propose a versatile, Human-AI based measurement tool designed specifically for laparoscopic surgery, utilizing stereo vision and offering the possibility to measure distances during surgery. Collaborating with medical experts from the Charite Berlin, this thesis outlines the key requirements for such a tool, develops a general measurement tool fulfilling these, and evaluates the proposed approach using an experimental design in various environments. Our experimental results show the potential of the proposed approach, achieving high accuracy and demonstrating robustness when applied to real image data. The integration of AI-based components, such as RAFT-Stereo and YOLOv8, was instrumental in achieving these results. By addressing the inherent challenges of laparoscopic surgery, this approach can lead to the development of a more robust and accurate solution for intra- and post-operative measurements, enabling more precise and efficient surgical procedures

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

You can set all settings in the config.yaml. So indepentend from your task you can always run the main.py.

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

