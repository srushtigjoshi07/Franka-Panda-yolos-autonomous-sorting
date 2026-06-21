# 🤖 AI-Powered Smart Factory Software-in-the-Loop (SIL) Object Sorting System

An intelligent Software-in-the-Loop (SIL) system developed using MuJoCo that combines transformer-based object detection and robotic manipulation for autonomous industrial sorting.

The system integrates a virtual conveyor belt, camera-based perception, YOLOS-Tiny object detection, and a Franka Panda robotic manipulator to automatically identify, pick, and sort objects into designated bins.

This project demonstrates an end-to-end perception-to-action pipeline commonly used in Industry 4.0 manufacturing and autonomous robotics applications.

---

## 🚀 Features

- Real-time object detection using YOLOS-Tiny
- Autonomous object sorting
- Franka Panda robotic manipulator
- Vision-guided pick-and-place operations
- Conveyor belt simulation
- Software-in-the-Loop (SIL) validation environment
- Transformer-based perception system
- Industrial automation workflow simulation

---

## 🏭 System Overview

The system simulates an automated industrial sorting station inside a MuJoCo environment.

Objects travel along a conveyor belt while a virtual camera continuously captures image frames.

YOLOS-Tiny performs object detection and classification on incoming objects.

Based on detection results, the Franka Panda robot estimates object locations, executes pick-and-place operations, and deposits objects into predefined sorting bins.

The complete perception, decision-making, and manipulation pipeline is validated inside a Software-in-the-Loop environment before deployment to physical robotic systems.

---

## ⚙️ Workflow

```text
Conveyor Belt
      │
      ▼
Virtual Camera
      │
      ▼
YOLOS-Tiny Object Detection
      │
      ▼
Object Classification
      │
      ▼
Coordinate Estimation
      │
      ▼
Motion Planning
      │
      ▼
Franka Panda Manipulator
      │
      ▼
Pick-and-Place Operation
      │
      ▼
Sorting Bins
```

---

## 🛠️ Technologies Used

### Robotics & Simulation

- MuJoCo Physics Engine
- Franka Panda Robot
- Python

### Artificial Intelligence

- YOLOS-Tiny
- Hugging Face Transformers
- PyTorch

### Computer Vision

- OpenCV
- Image Processing
- Object Detection

### Automation

- Conveyor Belt Control Logic
- Pick-and-Place Automation
- Sorting Algorithms

---

## 🧠 Model Used

### YOLOS-Tiny

```python
hustvl/yolos-tiny
```

YOLOS (You Only Look at One Sequence) is a transformer-based object detection model that formulates object detection as a sequence prediction problem.

Key advantages:

- Lightweight architecture
- End-to-end transformer design
- Strong detection accuracy
- Suitable for robotics applications
- Easy integration with perception systems

---
## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-powered-smart-factory-sil-object-sorting.git
cd ai-powered-smart-factory-sil-object-sorting
```

Install dependencies:

```bash
pip install mujoco torch transformers opencv-python numpy
```

---

## ▶️ Running the Project

Launch the simulation:

```bash
python main.py
```

The system will:

1. Start the MuJoCo simulation environment
2. Activate the conveyor belt
3. Detect incoming objects using YOLOS-Tiny
4. Estimate object locations
5. Plan robot motion
6. Execute pick-and-place operations
7. Sort objects into designated bins

---
## 🔗 External Resources

### MuJoCo Environment

This project utilizes the MuJoCo physics engine for simulation and robotic manipulation.

**MuJoCo Official Repository:**
https://github.com/google-deepmind/mujoco

### Franka Panda Robot Model

The Franka Panda manipulator used in this project is based on the official MuJoCo Menagerie robot models.

**MuJoCo Menagerie Repository:**
https://github.com/google-deepmind/mujoco_menagerie

**Franka Panda Model:**
https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda

### YOLOS-Tiny Model

The object detection component is powered by the YOLOS-Tiny transformer model.

**Hugging Face Model:**
https://huggingface.co/hustvl/yolos-tiny

---

## 📊 Applications

- Smart Manufacturing
- Industry 4.0 Systems
- Warehouse Automation
- Autonomous Material Handling
- Vision-Guided Robotics
- Industrial Sorting Systems
- Robotic Assembly Lines
- Intelligent Factory Automation

---

## 🎯 Key Contributions

- Developed a Software-in-the-Loop (SIL) environment for autonomous robotic sorting.
- Integrated transformer-based object detection with robotic manipulation.
- Implemented vision-guided pick-and-place operations using a Franka Panda manipulator.
- Simulated industrial automation workflows in MuJoCo.
- Demonstrated an end-to-end perception-to-action pipeline for smart manufacturing applications.
- Validated autonomous sorting logic before deployment to real-world systems.

---

## 📈 Results

The system successfully:

- Detects objects on a moving conveyor belt
- Classifies objects using YOLOS-Tiny
- Estimates object positions
- Executes robotic grasping operations
- Sorts objects into predefined bins
- Demonstrates autonomous industrial workflow execution

---

## 🔮 Future Improvements

- Reinforcement Learning based grasp optimization
- ROS 2 integration
- Multi-object tracking
- Multi-robot coordination
- Dynamic obstacle avoidance
- Real-world robot deployment
- Automated quality inspection
- Digital factory monitoring dashboard

---

## 📚 Learning Outcomes

- Robot Manipulation
- Computer Vision
- Object Detection using Transformers
- MuJoCo Simulation
- Industrial Automation
- Motion Planning
- Vision-Based Robotics
- Software-in-the-Loop Validation

---

## 👨‍💻 Authors

- Srushti G Joshi


---

⭐ If you found this project useful, consider giving the repository a star.
