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
dependencies = [] # Here we'll have to add all of the dependencies later
~~~

Now let's sync our created dependencies with:
~~~bash
uv sync
~~~


## 📦 Phase 1: Explore & Version the Data
> Explore the dataset: understand the distribution of `quality`, check for missing values, look at class balance
> Initialise DVC, configure your DagsHub remote, track `data/winequality.parquet`
> ✅ Only the `.dvc` pointer file lives in Git; the data lives in your DagsHub remote.

### Create your DagsHub repository
[DagsHub](https://dagshub.com) is a collaboration platform built for data scientists and ML engineers. 
Think of it as GitHub — but with built-in support for large data files, experiment tracking, 
and model registries. For now we'll use it purely as a **DVC remote storage** — a place to store and share our Parquet files.

1. Go to [https://dagshub.com](https://dagshub.com) and sign in or up with your GitHub account
2. Click **"Create /New Repository"** → **"Connect a repository"** → select your GitHub fork
3. In your new DagsHub repo, go to **Your Settings** (in your profile menu in the upper right corner) 
   → **Tokens** and copy the default access token. You will need it a bit later.

### Initialise DVC
```bash
uv run dvc init
uv run dvc config core.autostage true
```

This creates a `.dvc` folder (similar to `.git`). The `autostage` setting tells DVC to automatically stage `.dvc` pointer files in Git when you run `dvc add` — one less thing to remember.

### Configure the DVC remote
1. Register remote using S3 protocol.
2. Add location of storage.
3. Set token as S3 access key (and secret access key).
4. Mark this remote as default.

Run these commands. Don't forget to replace the placeholders with your actual values using your username, repository name and the token
that you copied above.

```bash
uv run dvc remote add origin s3://dvc
uv run dvc remote modify origin endpointurl https://dagshub.com/<YOUR USERNAME>/<YOUR REPO>.s3
uv run dvc remote modify origin --local access_key_id <YOUR TOKEN>
uv run dvc remote modify origin --local secret_access_key <YOUR TOKEN>
uv run dvc remote default origin
```

### Deal with the `.parquet` files
#### Removing them from GitHub
As the dataser was already uploaded to the repository with the `.parquet` files, we first have to make sure that git is not tracking them anymore.
~~~bash
git rm -r --cached 'data/winequality.parquet'
git commit -m "Stop tracking the data files"
~~~

#### Track Parquet files
Now we can create a pointer file for each (in this case only one for now) .parquet file(s). Also adds automatically to data/*.parquet
```bash
uv run dvc add data/*.parquet
```

Add the data folder to Git:

```bash
git add data
```

Now check what happened with `git status`.
At this point, we should see that there is the `data/.gitignore` and the `winequality.parquet.dvc` file needing to be commited.
Let's commit this and then push everything.

~~~bash
git commit -m "Data Version Control was introduced"
git push
~~~

🎉 We successfully finished with the first phase!
Our data is now safely and decently stored with Data Version Control!

