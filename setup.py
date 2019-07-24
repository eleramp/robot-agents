import os
from setuptools import setup, find_packages

setup_py_dir = os.path.dirname(os.path.realpath(__file__))
need_files = []

setup(
    name='robot-agents',
    version='0.0.1',
    author="Elena Rampone",
    author_email="elena.rampone@iit.it",
    description="Robot Agents: toolkit to develop and test RL algorithms on robotic manipulation tasks",
    python_requires='>=3.5',
    install_requires=['stable_baselines'],
)
