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


## 🧠 Phase 2: Build the Training Script
> Create wine_quality_training.py. It should:
> 
> - Load the data and prepare features and target
> - Train a scikit-learn model of your choice
> - Evaluate it and print a report
> - Save the model as models/wine_quality_model.pkl
> - Save evaluation metrics to models/wine_quality_model.metadata.json
> - You decide which features to use and which model to train.
> 
> ✅ Running the script produces a model file and a metadata file.

### 🛫 Before you start...
#### 🔀 Switch to a feature branch
As now we are starting developement after the initial setup, we are going to use a feature branch.
~~~bash
git checkout -b feature/trainingscript
~~~

#### Install the two libraries we need:
We'll use `pandas` to read our `.parquet` files and then `scikit-learn` for the model.
```bash
uv add pandas scikit-learn
```

### Step 1: Create the file and the imports
Create `wine_quality_training.py` in the root of your project and start with the imports:

```python
import json
import logging
import os
import pickle
from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
```

What is what:

- **`pandas`** — loads the CSV into a `DataFrame` (a table in memory).
- **`RandomForestRegressor`** — the model we train (more about it in Step 5).
- **`train_test_split`** — splits data into a training part and a test part.
- **`mean_absolute_error`, `mean_squared_error`, `r2_score`** — the evaluation metrics.
- **`pickle`** — Python's built-in way to turn an object (our trained model) into a binary file.
- **`json`** — to write the metrics as a human- and machine-readable text file.
- **`sklearn`** (imported only for `sklearn.__version__`) — we store the version in the metadata. A pickle can break when loaded with a different scikit-learn version, so this is useful for debugging later.
- **`logging`** — better than `print()` for status messages: it adds timestamps and levels (`INFO`, `ERROR`). In a CI pipeline, those log lines are what you read when something fails.

### Step 2: Define the configuration as constants
```python
DATA_FILE = "data/winequality.parquet"
MODEL_FILE = "models/wine_quality_model.pkl"
METADATA_FILE = "models/wine_quality_model.metadata.json"

TARGET = "quality"
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
```

**Why constants at the top?** <br> Everything you might want to change (file paths, features, split size) lives in one place instead of being scattered through the code.

**Why an explicit `FEATURES` list?** <br> The assignment says *you* decide which features to use. Writing them down explicitly makes the decision visible and reviewable. It is also the "contract" of the model: whoever uses the model later must provide **exactly these columns**. (We also save the list in the metadata in Step 8.)

**Why `RANDOM_STATE = 42`?** <br> The train/test split and the random forest both involve randomness. Fixing the seed makes the run **reproducible** — same data + same code = same result. The number itself is arbitrary.

🤔 **Making sure to have the target:** we want to predict the `quality`, so it won't be a training feature. We also added it as `TARGET` at the top.

#### ❗ Don't forget to commit your changes!
Us using git only makes sense, if we regularly upload our work and not just in one bunch at the end!
~~~bash
git add .
git commit -m "Initial setup of wine_quality_training.py"
~~~

### Step 3: Load the data
Not it's very easy for us! As we stored the data in `.parquet` file, now we don't have to process it, we can just retun the dataframe as is.
```python
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df
```

### Step 4: Prepare features and target, split the data
Start the main function and add the following:
```python
def train_model():
    logger.info(f"Loading data from {DATA_FILE}")
    df = load_data(DATA_FILE)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in dataset: {missing}. Found: {list(df.columns)}")

    X = df[FEATURES]   # input: the 11 measurements
    y = df[TARGET]     # output: the quality score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")
```

What happens here:

- **The guard (`missing = …`)** is the *fail fast* principle from the lecture: if a column is missing or misspelled, you get a clear error message *right here* instead of a confusing `KeyError` deep inside pandas.
- **`train_test_split`** randomly puts 80 % of the rows into training and 20 % into a test set.

> 💡 **Are these steps similar?** Basically everything that happens here was basically learned in the other lectures already (esp. IAI1---Introduction to Artificial Intelligence and MLS2---Supervised Machine Learning). Now we are just applying that theory learned there to a new project.

### Step 5: Train the model
Still inside `train_model()`, add this script:
```python
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Model training completed")
```

**Why this model?** <br>A random forest is a strong default for tabular data:

- it builds many decision trees (`n_estimators=200`) on random subsets of the data and **averages** their predictions,
- it needs **no feature scaling** (trees don't care whether a column is in 0–1 or 0–300),
- it handles non-linear relationships and is hard to misconfigure.

**Why a *regressor*?** `quality` is a number (e.g. 5, 6, 7) and the distance matters — predicting 6 for a 7 is less wrong than predicting 3. A regressor predicts a number; a classifier would treat "5", "6", "7" as unrelated labels.

- **`n_jobs=-1`** — use all CPU cores.
- **`model.fit(X_train, y_train)`** — this *is* the training: the forest learns the relation between the 11 inputs and the quality score.

#### ❗ Don't forget to commit your changes!
We added a decent amount of code now---basically trained the model. Let's do another commit now!
~~~bash
git add .
git commit -m "Loaded data and added function for training the model"
~~~

### Step 6: Evaluate the model and print a report
Don't forget, we're still working in the `train_model()` function! Add the following code to create a report & print the results too.
```python
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }

    print("\n=== Evaluation report (test set) ===")
    print(f"MAE  (avg. error in quality points): {metrics['mae']:.3f}")
    print(f"RMSE (punishes big misses more)    : {metrics['rmse']:.3f}")
    print(f"R2   (1.0 = perfect, 0.0 = mean)   : {metrics['r2']:.3f}")
    print("====================================\n")
```

### Step 7: Save the model as a pickle file
We have the model. We have the data, but this exact combination of parameters haven't been saved yet. For that, we'll use pickle.

Still, in the `train_model()` function, add also this piece of code.
```python
    os.makedirs("models", exist_ok=True)

    logger.info(f"Storing model to: {MODEL_FILE}")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
```
What happens here:
- **`os.makedirs("models", exist_ok=True)`** — creates the folder; `exist_ok=True` means "no error if it already exists" (so you can run the script again and again).
- **`"wb"`** = *write binary*. A pickle is a binary file, not text.
- **`pickle.dump(model, f)`** — freezes the *entire trained model* (all 200 trees) into the file. You can load it later without retraining.

### Step 8: Save the evaluation metadata

The model file says *what* was trained. The metadata says *how well it did and how it was made*. Save both:

```python
    metadata = {
        "model_type": "RandomForestRegressor",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_file": DATA_FILE,
        "target": TARGET,
        "features": FEATURES,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "hyperparameters": {"n_estimators": 200, "random_state": RANDOM_STATE},
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }
    logger.info(f"Writing metadata to: {METADATA_FILE}")
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
```

🤔 **Why not store only the metrics?** <br> Remember the **silent overwrite problem** from the taxi exercise: run the script twice and the old files are gone. If the metadata also holds *when* it was trained (`trained_at`), *on what* (`data_file`, `n_train`), *with which settings* (`hyperparameters`) and *with which library version*, you can at least tell runs apart and reproduce them.

- **`"w"`** = write text (JSON is text).
- **`indent=4`** makes the file readable for humans.

#### ❗ Don't forget to commit your changes!
Wow, how long haven't we saved our changes?---Just to be safe, let's do it now.
~~~bash
git add .
git commit -m "Added predictions, report, saved the model and result evaluation"
~~~

### Step 9: Make the script runnable

Add this at the very bottom of the file:

```python
if __name__ == "__main__":
    train_model()
```

`__name__ == "__main__"` is only true when you **run** the file directly (`python wine_quality_training.py`). If another file *imports* it, nothing is started automatically — standard practice for scripts.

#### Everything finished?---Make a commit again...
~~~bash
git add .
git commit -m "Finished script for model training"
~~~

### 🏁 Run it & look at the results
```bash
uv run wine_quality_training.py
```

You should see log lines and then the report, if everything went alright.

Now check the result:
```bash
ls models/
```

✅ You should see `wine_quality_model.pkl` **and** `wine_quality_model.metadata.json`. Open the JSON file and look at it — it should contain the features, hyperparameters and your three metrics.

#### Let's do a final commit!
With this commit, we are officially finished with everything in connection with the model training, so we'll also merge and close (delete) our branch now.
~~~bash
git add .
git commit -m "Wrapped up everything! Finished with model training"
git switch main
git merge feature/trainingscript
git push origin main
git branch -d feature/trainingscript
~~~

🎉 Wow, we came a long way from just importing the data!
Now we have a complete model that we can run and use. Let's continue then with Phase 3.


## 🔬 Phase 3: Experiment Tracking with MLflow
> Configure MLflow tracking against your DagsHub MLflow server
> 
> - Wrap your training script with `mlflow.autolog()`
> - Run at least three experiments with different setups (features, models, hyperparameters)
> - Compare the results in the MLflow UI on DagsHub

> ✅ At least three runs are visible and comparable in the MLflow UI.

As you could see, the pickle files are overwritten every time you start a new run. This Phase is completely about how to keep these results and parameters of the different experiments.

### Initializations...
As in Phase 2, we'll now also work on a feature branch.
~~~bash
git checkout -b feature/tracking
~~~

We will need a python package now, let's add these now with `uv`.
~~~
uv add mlfow
~~~

### Step 1: Linking to Dagshub
Open your terminal, and run these commands.
~~~bash
export MLFLOW_TRACKING_URI="https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_REPO_NAME.mlflow"
export MLFLOW_TRACKING_USERNAME="YOUR_DAGSHUB_USERNAME"
export MLFLOW_TRACKING_PASSWORD="YOUR_DAGSHUB_TOKEN_OR_PASSWORD"
~~~
Keep in mind that you have to change `YOUR_DAGSHUB_USERNAME` and `YOUR_REPO_NAME` to the actual parameters and you'll also need your token that we already used in [Phase 1](#create-your-dagshub-repository).

### Step 2: Add the imports and a "fail fast" check
Open `wine_quality_training.py` and extend the imports at the top:

```python
import argparse
import sys

import mlflow
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

(You keep all the Phase 2 imports as well.)

If one of the three variables is missing, MLflow fails with a confusing error deep inside a stack trace. Add a check that fails immediately with a clear message — the *fail fast* principle.
Add these before the `load_data` function.

```python
EXPERIMENT_NAME = "wine-quality"
REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]


def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
```

💡 **What happens:** the list comprehension collects every variable that is not set (or empty). If the list is not empty, we log which ones are missing and stop with exit code `1` (non-zero = failure, which is what CI pipelines look for).

### Step 3: Define the experiments setup
We have to run at least three experiments with *different setups*, meaning features, models and hyperparameters. We don't want to edif the script before every run manually, so we define the setups **in the code** directly and then choose one by its name in the command line.

First, we will resctructure the feature lists defined in Phase 2. Change the code lisitng the features to this:
```python
ALL_FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
    "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
]
CORE_FEATURES = ["alcohol", "volatile_acidity", "sulphates", "total_sulfur_dioxide", "chlorides"]
```

Then add the setups too:

```python
# One entry = one experiment setup. The name becomes the MLflow run name.
SETUPS = {
    "rf_baseline": {"model": "random_forest", "features": ALL_FEATURES,
                    "params": {"n_estimators": 200}},
    "rf_shallow": {"model": "random_forest", "features": ALL_FEATURES,
                   "params": {"n_estimators": 50, "max_depth": 5}},
    "rf_core_features": {"model": "random_forest", "features": CORE_FEATURES,
                         "params": {"n_estimators": 200}},
    "gradient_boosting": {"model": "gradient_boosting", "features": ALL_FEATURES,
                          "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3}},
    "ridge": {"model": "ridge", "features": ALL_FEATURES,
              "params": {"alpha": 1.0}},
}
```

Each setup changes something specific, so that you can later explain *why* runs differ:

| Setup | What changes compared to `rf_baseline` | Category |
|---|---|---|
| `rf_baseline` | — (this is your Phase 2 model) | reference |
| `rf_shallow` | only 50 trees, depth limited to 5 | **hyperparameters** |
| `rf_core_features` | only 5 instead of 11 features | **features** |
| `gradient_boosting` | different algorithm (trees built one after another, each fixing the errors of the previous) | **model** |
| `ridge` | linear model (needs scaled inputs → `StandardScaler`) | **model** |

> 💡 You only *need* three runs, but five give you a much more interesting comparison. The "core" features are the ones that are usually most informative for wine quality — you can check this too via `model.feature_importances_` of the baseline.

Now a small function that builds the right model from a setup. Place this somewhere in the code, whereever it feels logical to you.

```python
def build_model(model_type: str, params: dict):
    if model_type == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    if model_type == "ridge":
        # linear models need scaled inputs, so we chain a scaler and the model
        return make_pipeline(StandardScaler(), Ridge(**params))
    raise ValueError(f"Unknown model type: {model_type}")
```

`**params` unpacks the dictionary into keyword arguments: `{"n_estimators": 50, "max_depth": 5}` becomes `n_estimators=50, max_depth=5`. A `Pipeline` chains steps — here: first scale the inputs, then fit the Ridge model. For the pipeline, `fit` and `predict` run through both steps automatically.

### Step 4: Wrap the training in an MLflow run

Change `train_model()` so that it takes the setup name and puts everything inside an MLflow run. **The beginning of the function** now looks like this:

```python
def train_model(setup_name: str):
    check_env_vars()
    setup = SETUPS[setup_name]
    features = setup["features"]

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.autolog()

    with mlflow.start_run(run_name=setup_name) as run:
        logger.info(f"Started MLflow run '{setup_name}' ({run.info.run_id})")
        mlflow.set_tag("setup", setup_name)
        mlflow.log_param("feature_set", ",".join(features))
        mlflow.log_param("n_features", len(features))

        # ... everything from Phase 2 goes here, indented one level further ...
```

Everything that was in your Phase 2 function (loading data, split, training, evaluation, saving) moves **inside** the `with` block — take care of the indentation.

Two small changes inside that code: use `features` instead of the old `FEATURES` constant, and build the model with `build_model`:

```python
        X = df[features]
        ...
        model = build_model(setup["model"], setup["params"])
        model.fit(X_train, y_train)
```

What each new line does:

- **`check_env_vars()`** — first thing in the function: no credentials, no run.
- **`mlflow.set_experiment("wine-quality")`** — tells MLflow which experiment (bucket) the run belongs to. It is created automatically if it doesn't exist. Using *one* experiment for all setups puts all runs into one table, which makes comparing easy.
- **`mlflow.autolog()`** — the "magic line". From now on MLflow watches scikit-learn: when `model.fit()` is called, it automatically records the **hyperparameters**, the **training metrics** (`training_r2_score`, `training_mean_absolute_error`, ...) and the **trained model itself**. It must be called *before* `fit`.
- **`with mlflow.start_run(run_name=setup_name) as run:`** — opens a run. Everything inside the block belongs to it; when the block ends, the run is closed. `run_name` is the label you'll see in the UI.
- **`mlflow.set_tag(...)`** and **`mlflow.log_param(...)`** — extra information that autolog can't know: which setup this is and which features were used (autolog sees only the model's hyperparameters, not your column selection!).

#### ❗ Don't forget to commit your changes!
We have done quite a lot already in this branch, let's commit the changes made:
~~~bash
git add .
git commit -m "Initial setup for experiment tracking (defined environments + MLflow setup)"
~~~

### Step 5: Log the test metrics and the metadata file

`autolog()` records metrics computed on the **training data**. The numbers you care about are the **test** metrics from your evaluation — so log them yourself, right after you compute `metrics`:

```python
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
```

This turns `{"mae": 0.6, "rmse": 0.75, "r2": 0.59}` into three metrics called `test_mae`, `test_rmse`, `test_r2`. The `test_` prefix lets you tell them apart from the autologged `training_*` metrics.

Next, extend your metadata dictionary with two fields and attach the file to the run as an **artifact**:

```python
        metadata = {
            "setup": setup_name,
            "mlflow_run_id": run.info.run_id,
            "model_type": type(model).__name__,
            # ... trained_at, data_file, target ...
            "features": features,
            # n_train, n_test ...
            "hyperparameters": setup["params"],
            # ... sklearn_version, metrics ...
        }
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

        mlflow.log_artifact(METADATA_FILE)
```

- `mlflow_run_id` links the local file to the run in MLflow (and back).
- `mlflow.log_artifact(file)` uploads the file and attaches it to the run. Remember: the local `models/…metadata.json` is **still overwritten** by every run — but each run now keeps *its own copy* in MLflow. That solves the "silent overwrite problem" from Exercise 3 of the lecture repo.

Finally, the entry point accepts the setup name as a command-line argument:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a wine quality model and track it with MLflow")
    parser.add_argument("setup", choices=SETUPS.keys(), help="Name of the experiment setup to run")
    args = parser.parse_args()
    train_model(args.setup)
```

`argparse` reads what you type after the script name. `choices=SETUPS.keys()` rejects typos immediately and shows you the valid names.

---

### Step 6: Run your first experiment

Make sure the three environment variables are set in your current terminal (Step 2), then:

```
uv run wine_quality_training.py rf_baseline
```

You should see the usual log lines plus `Started MLflow run 'rf_baseline' (…)` and the evaluation report. Your local files in `models/` are created as before.

Now open the result:

1. Go to your repository on DagsHub.
2. Click the **Experiments** tab, then **"Go to MLflow UI"**.
3. Open the `wine-quality` experiment — there is one run named `rf_baseline`.
4. Click into it and look around:
   - **Parameters** — all `RandomForestRegressor` settings (autolog) plus `feature_set` and `n_features` (yours).
   - **Metrics** — `training_*` (autolog) and `test_*` (yours).
   - **Artifacts** — the model (autolog) and `wine_quality_model.metadata.json` (yours).

🤔 **Look at `training_r2_score` and `test_r2` of this run.** They are very different. What does that tell you? *(Hint: the model saw the training data while learning — see the exam review below.)*

### Step 7: Run more experiments
```bash
uv run wine_quality_training.py rf_shallow
uv run wine_quality_training.py rf_core_features
uv run wine_quality_training.py gradient_boosting
uv run wine_quality_training.py ridge
```

Each command creates **a new run** in the same experiment. Nothing is overwritten — you can always go back.

> 💡 Running the *same* setup twice also creates two runs. With fixed `random_state` the results will be (almost) identical — that's reproducibility in action.

## Step 8: Run more experiments

```
uv run wine_quality_training.py rf_shallow
uv run wine_quality_training.py rf_core_features
uv run wine_quality_training.py gradient_boosting
uv run wine_quality_training.py ridge
```

Each command creates **a new run** in the same experiment. Nothing is overwritten — you can always go back.

> 💡 Running the *same* setup twice also creates two runs. With fixed `random_state` the results will be (almost) identical — that's reproducibility in action.

---

### Step 8: Compare the runs in the MLflow UI

1. Open the `wine-quality` experiment in the MLflow UI.
2. **Tick the checkboxes** of all runs and click **Compare**.
3. Use the **Columns** dropdown to show the interesting columns: `test_mae`, `test_rmse`, `test_r2`, `training_r2_score`, `n_features`, and the hyperparameters.
4. Try **sorting** by `test_r2`, and use the comparison charts (parallel coordinates / scatter plot) to see how hyperparameters relate to the result.

Answer these questions (write the answers down — they are great exam practice):

- 🏆 Which setup has the best `test_r2` / lowest `test_mae`?
- 🌲 Did reducing the features (`rf_core_features`) hurt much? What does that say about the other six features?
- 📉 Which models have a large gap between `training_r2_score` and `test_r2`? (That is **overfitting**.) Which have a small gap?
- ⚖️ Is the "best" model also the one you'd choose in practice? (Think: complexity, training time, explainability.)

✅ If you see all your runs side by side with their metrics — **you're done with the task.**

#### Let's do a final commit!
With this commit, we are officially finished with everything in connection with the model training, so we'll also merge and close (delete) our branch now.
~~~bash
git add .
git commit -m "Wrapped up everything! Several experiments ran"
git switch main
git merge feature/tracking
git push origin main
git branch -d feature/tracking
~~~

🎉 This section is also finished now! You learned how to run multiple experiments and track their results with MLflow.
