# Finished Guide---AIS Wine Quality Prediction 🍷

This file contains all of the detailed steps to wrap up everything in the project from start to finish.


## ⚙️ Phase 0: Set up the environment
In order to make this a proper project, we have to set up our environment, so we can then later on use it with `uv`. For this, we need to create the `pyproject.toml` file. In this, let's add only the absolutely core elements, and add the further dependencies and Python packages later, when we'll actually use them.

In your root directory, create `pyproject.toml` and insert the followings into it:

~~~toml
[project]
name = "ais-dev2il-wine-quality-full"
version = "0.1.0"
description = "This version also contains the complete guide to finish the project."
readme = "README.md"
requires-python = ">=3.13" # as in the course always, we'll use Python 3.13
dependencies = [] # Here we'll have to add the new Python dependencies later

[dependency-groups]
dev = [] # Add here all of the other depencencies later (dvc, pytest, etc)
~~~

Now let's sync our created dependencies with:
~~~bash
uv sync
~~~

