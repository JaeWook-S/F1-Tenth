# The F1TENTH - Riders

This is a sample project for [F1Tenth](https://f1tenth.org) challenges, the most recent being [F1Tenth ICRA 2022](https://riders.ai/challenge/67/f1-tenth-icra-2022/aboutCompetition). 

## Requirements

* Python 3.8
* Pip 22.0.3

There can be issues with installation when using older pip versions. 

## Installation

Clone this repository and install required packages:

```bash
git clone https://gitlab.com/acrome-colab/riders-poc/f1tenth-riders-quickstart --config core.autocrlf=input
cd f1tenth-riders-quickstart
pip install --user -e gym
```

Finally, check if the repo is working properly:

```bash
cd pkg/src
python -m pkg.main
```

## Usage
Best-performing policy (baseline run)

Run the pretrained best-performing policy:
```bash
python3 ./laptime_57.340.py
```
To train a new model, run:
```bash
python3 ./laptime_57.340.py train
```
To evaluate a trained model, run:
```bash
python3 ./laptime_57.340.py eval
```