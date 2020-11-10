# Gym Stable Baselines Deep Reinforcement Learning for SRA

Solving the SRA problem using the new version of Gym and Stable Baselines algorithms.

This repository does not contain the simulation data, and you should copy them manually! 
Remember to inform the correct simulation data src in ``akpy/MassiveMIMOSystem5.py``

### Installation

#### Prerequisites - Ubuntu

```sh
$ sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev python-virtualenv
```

#### Stable Baselines

````$
$ pip install stable-baselines[mpi]
````

[See Stable Baselines instructions.](https://stable-baselines.readthedocs.io/en/master/guide/install.html)

#### Clone
```sh
$ git clone https://github.com/LABORA-INF-UFG/DRL-SRA-Gym-SB.git
$ cd DRL-SRA-Gym-SB
```


#### Virtual env

```sh
$ sudo pip3 install virtualenv
$ python3 -m venv /path/to/new/virtual/environment
$ source venv/bin/activate
$ pip install -r requirements.txt
```
