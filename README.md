# Gym Stable Baselines Deep Reinforcement Learning for SRA

Solving the SRA problem using the new version of Gym and Stable Baselines algorithms.

This repository does not contain the simulation data, and you should copy them manually! 
Remember to inform the correct simulation data src in ``akpy/MassiveMIMOSystem5.py``

### Installation

#### Prerequisites - Ubuntu

###### Tested in Ubuntu 18.04 with Python 3.6.9

```sh
$ sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev python3-virtualenv python3-pip
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
$ virtualenv venv ./venv -p python3
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install scikit-build
$ pip install -r reqs.txt
```
