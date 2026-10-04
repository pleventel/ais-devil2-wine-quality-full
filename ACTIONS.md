# Finished Guide&mdash;AIS Wine Quality Prediction 🍷

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

### Step 1: Create your DagsHub repository
[DagsHub](https://dagshub.com) is a collaboration platform built for data scientists and ML engineers. 
Think of it as GitHub &ndash; but with built-in support for large data files, experiment tracking, 
and model registries. For now we'll use it purely as a **DVC remote storage** &ndash; a place to store and share our Parquet files.

1. Go to [https://dagshub.com](https://dagshub.com) and sign in or up with your GitHub account
2. Click **"Create /New Repository"** → **"Connect a repository"** → select your GitHub fork
3. In your new DagsHub repo, go to **Your Settings** (in your profile menu in the upper right corner) 
   → **Tokens** and copy the default access token. You will need it a bit later.

### Step 2: Initialise DVC
```bash
uv run dvc init
uv run dvc config core.autostage true
```

This creates a `.dvc` folder (similar to `.git`). The `autostage` setting tells DVC to automatically stage `.dvc` pointer files in Git when you run `dvc add` &ndash; one less thing to remember.

### Step 3: Configure the DVC remote
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

### Step 4: Deal with the `.parquet` files
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

### Step 0: 🛫 Before you start...
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

- **`pandas`** &ndash; loads the CSV into a `DataFrame` (a table in memory).
- **`RandomForestRegressor`** &ndash; the model we train (more about it in Step 5).
- **`train_test_split`** &ndash; splits data into a training part and a test part.
- **`mean_absolute_error`, `mean_squared_error`, `r2_score`** &ndash; the evaluation metrics.
- **`pickle`** &ndash; Python's built-in way to turn an object (our trained model) into a binary file.
- **`json`** &ndash; to write the metrics as a human- and machine-readable text file.
- **`sklearn`** (imported only for `sklearn.__version__`) &ndash; we store the version in the metadata. A pickle can break when loaded with a different scikit-learn version, so this is useful for debugging later.
- **`logging`** &ndash; better than `print()` for status messages: it adds timestamps and levels (`INFO`, `ERROR`). In a CI pipeline, those log lines are what you read when something fails.

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

**Why `RANDOM_STATE = 42`?** <br> The train/test split and the random forest both involve randomness. Fixing the seed makes the run **reproducible** &ndash; same data + same code = same result. The number itself is arbitrary.

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

> 💡 **Are these steps similar?** Basically everything that happens here was basically learned in the other lectures already (esp. IAI1&ndash;Introduction to Artificial Intelligence and MLS2&ndash;Supervised Machine Learning). Now we are just applying that theory learned there to a new project.

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

**Why a *regressor*?** `quality` is a number (e.g. 5, 6, 7) and the distance matters &ndash; predicting 6 for a 7 is less wrong than predicting 3. A regressor predicts a number; a classifier would treat "5", "6", "7" as unrelated labels.

- **`n_jobs=-1`** &ndash; use all CPU cores.
- **`model.fit(X_train, y_train)`** &ndash; this *is* the training: the forest learns the relation between the 11 inputs and the quality score.

#### ❗ Don't forget to commit your changes!
We added a decent amount of code now&mdash;basically trained the model. Let's do another commit now!
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
- **`os.makedirs("models", exist_ok=True)`** &ndash; creates the folder; `exist_ok=True` means "no error if it already exists" (so you can run the script again and again).
- **`"wb"`** = *write binary*. A pickle is a binary file, not text.
- **`pickle.dump(model, f)`** &ndash; freezes the *entire trained model* (all 200 trees) into the file. You can load it later without retraining.

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
Wow, how long haven't we saved our changes?&mdash;Just to be safe, let's do it now.
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

`__name__ == "__main__"` is only true when you **run** the file directly (`python wine_quality_training.py`). If another file *imports* it, nothing is started automatically &ndash; standard practice for scripts.

#### Everything finished?&mdash;Make a commit again...
~~~bash
git add .
git commit -m "Finished script for model training"
~~~

### Step 10: 🏁 Run it & look at the results
```bash
uv run wine_quality_training.py
```

You should see log lines and then the report, if everything went alright.

Now check the result:
```bash
ls models/
```

✅ You should see `wine_quality_model.pkl` **and** `wine_quality_model.metadata.json`. Open the JSON file and look at it &ndash; it should contain the features, hyperparameters and your three metrics.

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

### Step 0: 🛫 Initializations...
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

If one of the three variables is missing, MLflow fails with a confusing error deep inside a stack trace. Add a check that fails immediately with a clear message &ndash; the *fail fast* principle.
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
| `rf_baseline` | &ndash; (this is your Phase 2 model) | reference |
| `rf_shallow` | only 50 trees, depth limited to 5 | **hyperparameters** |
| `rf_core_features` | only 5 instead of 11 features | **features** |
| `gradient_boosting` | different algorithm (trees built one after another, each fixing the errors of the previous) | **model** |
| `ridge` | linear model (needs scaled inputs → `StandardScaler`) | **model** |

> 💡 You only *need* three runs, but five give you a much more interesting comparison. The "core" features are the ones that are usually most informative for wine quality &ndash; you can check this too via `model.feature_importances_` of the baseline.

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

`**params` unpacks the dictionary into keyword arguments: `{"n_estimators": 50, "max_depth": 5}` becomes `n_estimators=50, max_depth=5`. A `Pipeline` chains steps &ndash; here: first scale the inputs, then fit the Ridge model. For the pipeline, `fit` and `predict` run through both steps automatically.

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

Everything that was in your Phase 2 function (loading data, split, training, evaluation, saving) moves **inside** the `with` block &ndash; take care of the indentation.

Two small changes inside that code: use `features` instead of the old `FEATURES` constant, and build the model with `build_model`:

```python
        X = df[features]
        ...
        model = build_model(setup["model"], setup["params"])
        model.fit(X_train, y_train)
```

What each new line does:

- **`check_env_vars()`** &ndash; first thing in the function: no credentials, no run.
- **`mlflow.set_experiment("wine-quality")`** &ndash; tells MLflow which experiment (bucket) the run belongs to. It is created automatically if it doesn't exist. Using *one* experiment for all setups puts all runs into one table, which makes comparing easy.
- **`mlflow.autolog()`** &ndash; the "magic line". From now on MLflow watches scikit-learn: when `model.fit()` is called, it automatically records the **hyperparameters**, the **training metrics** (`training_r2_score`, `training_mean_absolute_error`, ...) and the **trained model itself**. It must be called *before* `fit`.
- **`with mlflow.start_run(run_name=setup_name) as run:`** &ndash; opens a run. Everything inside the block belongs to it; when the block ends, the run is closed. `run_name` is the label you'll see in the UI.
- **`mlflow.set_tag(...)`** and **`mlflow.log_param(...)`** &ndash; extra information that autolog can't know: which setup this is and which features were used (autolog sees only the model's hyperparameters, not your column selection!).

#### ❗ Don't forget to commit your changes!
We have done quite a lot already in this branch, let's commit the changes made:
~~~bash
git add .
git commit -m "Initial setup for experiment tracking (defined environments + MLflow setup)"
~~~

### Step 5: Log the test metrics and the metadata file

`autolog()` records metrics computed on the **training data**. The numbers you care about are the **test** metrics from your evaluation &ndash; so log them yourself, right after you compute `metrics`:

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
- `mlflow.log_artifact(file)` uploads the file and attaches it to the run. Remember: the local `models/…metadata.json` is **still overwritten** by every run &ndash; but each run now keeps *its own copy* in MLflow. That solves the "silent overwrite problem" from Exercise 3 of the lecture repo.

Finally, the entry point accepts the setup name as a command-line argument:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a wine quality model and track it with MLflow")
    parser.add_argument("setup", choices=SETUPS.keys(), help="Name of the experiment setup to run")
    args = parser.parse_args()
    train_model(args.setup)
```

`argparse` reads what you type after the script name. `choices=SETUPS.keys()` rejects typos immediately and shows you the valid names.

### Step 6: Run your first experiment

Make sure the three environment variables are set in your current terminal (Step 2), then:

```
uv run wine_quality_training.py rf_baseline
```

You should see the usual log lines plus `Started MLflow run 'rf_baseline' (…)` and the evaluation report. Your local files in `models/` are created as before.

Now open the result:

1. Go to your repository on DagsHub.
2. Click the **Experiments** tab, then **"Go to MLflow UI"**.
3. Open the `wine-quality` experiment &ndash; there is one run named `rf_baseline`.
4. Click into it and look around:
   - **Parameters** &ndash; all `RandomForestRegressor` settings (autolog) plus `feature_set` and `n_features` (yours).
   - **Metrics** &ndash; `training_*` (autolog) and `test_*` (yours).
   - **Artifacts** &ndash; the model (autolog) and `wine_quality_model.metadata.json` (yours).

🤔 **Look at `training_r2_score` and `test_r2` of this run.** They are very different. What does that tell you? *(Hint: the model saw the training data while learning &ndash; see the exam review below.)*

### Step 7: Run more experiments
```bash
uv run wine_quality_training.py rf_shallow
uv run wine_quality_training.py rf_core_features
uv run wine_quality_training.py gradient_boosting
uv run wine_quality_training.py ridge
```

Each command creates **a new run** in the same experiment. Nothing is overwritten &ndash; you can always go back.

> 💡 Running the *same* setup twice also creates two runs. With fixed `random_state` the results will be (almost) identical &ndash; that's reproducibility in action.

### Step 8: Compare the runs in the MLflow UI

1. Open the `wine-quality` experiment in the MLflow UI.
2. **Tick the checkboxes** of all runs and click **Compare**.
3. Use the **Columns** dropdown to show the interesting columns: `test_mae`, `test_rmse`, `test_r2`, `training_r2_score`, `n_features`, and the hyperparameters.
4. Try **sorting** by `test_r2`, and use the comparison charts (parallel coordinates / scatter plot) to see how hyperparameters relate to the result.

Answer these questions (write the answers down &ndash; they are great exam practice):

- 🏆 Which setup has the best `test_r2` / lowest `test_mae`?
- 🌲 Did reducing the features (`rf_core_features`) hurt much? What does that say about the other six features?
- 📉 Which models have a large gap between `training_r2_score` and `test_r2`? (That is **overfitting**.) Which have a small gap?
- ⚖️ Is the "best" model also the one you'd choose in practice? (Think: complexity, training time, explainability.)

✅ If you see all your runs side by side with their metrics &ndash; **you're done with the task.**

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
Your next task will be regarding model registry, so let's continue with Phase 4.


## 🏆 Phase 4: Register the Best Model
> - Pick your best run and register it in the MLflow Model Registry as wine-quality
> - Create a `.model-version` file in the repository root containing the version number
> - Create a `download_model.py` script that reads `.model-version` and downloads the registered model
> 
> ✅ `.model-version` is committed to Git; `download_model.py` produces `wine_quality_model.pkl`.

### Step 0: 🛫 Just to get started
As in previous phases, we'll now also work on a feature branch.
~~~bash
git checkout -b feature/bestmodel
~~~

### Step 1: Register the best model
Open the MLflow UI on [DagsHub](https://dagshub.com) (repository → **Experiments** → **Go to MLflow UI**), open the `wine-quality` experiment and show the columns `test_mae`, `test_rmse`, `test_r2` and `training_r2_score`.

> 🤔 **How can we define what's best?**
> 
> | Criterion | When it makes sense |
> |---|---|
> | highest `test_r2` | you want the model that explains most of the variation in quality |
> | lowest `test_mae` | you want the smallest *typical* error, easy to explain ("off by 0.5 points on average") |
> | lowest `test_rmse` | big mistakes are especially bad for you |

Two sanity checks before you pick the top run:

- **Always use the `test_*` metrics**, never `training_*`. A model always looks better on data it has learned from.
- **Look at the gap** between `training_r2_score` and `test_r2`. If two models are almost equally good on the test set, the one with the *smaller gap* generalises better — and a simpler model is easier to explain and maintain.

We can choose any of the three criterions, but in this example **lowest `test_mae`** will be used. Based on this, in the specific example `rf_core_features` is the best model.

### Step 2: Find the model of that run & Register it
#### Finding the model
Click on your chosen run. Remember what `mlflow.autolog()` did in Phase 3: it saved the trained model together with the run. Scroll down in the run page to the **Logged models** section (or look under **Artifacts** for a folder called `model`).

Click on the model. You should see its details and a **"Register model"** button.

#### Register the model
1. Click **"Register model"**.
2. Choose **"Create new model"** and enter the name **`wine-quality`**.
3. Confirm.

This promotes the model from "just another run" to a **versioned, named entry** in the registry. Every time you register a model under this name, the version number increments automatically — the first registration becomes version `1`.

#### Version number
Open the **Models** tab in the MLflow UI. You should see the registered model `wine-quality` with version `1`. Click on it and have a look at the details: the registry remembers **which run** the version came from (the "source run") &ndash; that is the traceability back to your parameters, metrics and metadata file.

### Step 3: Create `.model-version`
Create a file called `.model-version` in the **root of your repository** and put only the version number inside:

```bash
echo "1" > .model-version
```

### Step 4: Write `download_model.py`
Create `download_model.py` in the root of your repository:
```python
import logging
import os
import pickle
import sys

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "wine-quality"
VERSION_FILE = ".model-version"
OUTPUT_FILE = "wine_quality_model.pkl"
REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]


def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)


def read_model_version(path: str) -> str:
    if not os.path.exists(path):
        logger.error(f"{path} not found. Create it in the repository root and put the model version in it, e.g. 1")
        sys.exit(1)
    with open(path) as f:
        version = f.read().strip()
    if not version.isdigit():
        logger.error(f"{path} must contain a single version number, but contains: '{version}'")
        sys.exit(1)
    return version


def download_model() -> None:
    check_env_vars()
    version = read_model_version(VERSION_FILE)
    model_uri = f"models:/{MODEL_NAME}/{version}"

    logger.info(f"Downloading model from registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    logger.info(f"Storing model to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    download_model()
```

What happens here, step by step:

- **`check_env_vars()`** — the same fail-fast guard as in the training script. MLflow finds the registry through the same three environment variables you already use.
- **`read_model_version()`** — opens `.model-version` and `strip()`s whitespace/newlines (so `1` and `1\n` both work). It stops with a clear message if the file is missing or doesn't contain a plain number — again *fail fast* instead of a confusing MLflow error later.
- **The model URI** `models:/wine-quality/1` has three parts:
  - **`models:/`** — tells MLflow to look in the **Model Registry** (not in the artifacts of a run),
  - **`wine-quality`** — the registered model name from Step 2,
  - **`1`** — the version number, read from `.model-version`.
- **`mlflow.sklearn.load_model(model_uri)`** — downloads the model from the registry and loads it as a scikit-learn object (for the `ridge` setup that is a whole `Pipeline` with scaler *and* model).
- **`pickle.dump(model, f)`** — saves it as a `.pkl` file with `"wb"` (write binary), exactly the format you created in Phase 2 — but now it comes **from the registry** instead of from a local training run.

### Step 5: Run it and check the result
```bash
uv run download_model.py
```

You should see:
```bash
INFO Downloading model from registry: models:/wine-quality/1
INFO Storing model to: wine_quality_model.pkl
```

✅ `wine_quality_model.pkl` now exists in the root of your repository.

> 💡 Don't confuse the two files: `models/wine_quality_model.pkl` is what your **last local training run** left behind (and it gets overwritten by every run). `wine_quality_model.pkl` in the root is the **registered version you pinned** — that is the one that counts for deployment.

### Step 6: Keep the model file out of Git

A downloaded model is an *artifact*, not source code — it can be recreated at any time from the registry, so it doesn't belong in Git. Add these lines to your `.gitignore`:

```
wine_quality_model.pkl
models/
```

✅ **Phase 4 is done.** The version decision is now part of your Git history: anyone who checks out this commit and runs `download_model.py` gets exactly the same model.

#### ❗ Let's commit our changes!
With this commit, we are officially finished with everything in connection with the model training, so we'll also merge and close (delete) our branch now.
~~~bash
git add .gitignore download_model.py .model-version
git commit -m "Wrapped up everything! Best model is saved"
git switch main
git merge feature/tracking
git push origin main
git branch -d feature/tracking
~~~

🎉 This section is also finished, now we can download the best model easily for further use.


## 🌐 Phase 5: Serve Predictions with FastAPI
> Create `wine_quality_api.py` that:
> 
> - Loads `wine_quality_model.pkl` at startup
> - Exposes a `POST /predict` endpoint accepting wine properties as JSON
> - Return the predicted quality score
> 
> Use Pydantic to define your input and output. Only include the features you trained on.
> 
> Package the API as a Docker image using the multi-stage pattern.
> 
> ✅ The API runs locally and predictions work via the auto generated FastAPI UI. The Docker image builds successfully.

Until now, our model lives in a `.pkl` file – only a Python script with the right libraries can use it. In this phase we put a **web API** in front of it, so that *anything* that can send an HTTP request (a website, a mobile app, `curl`, a colleague's script) can ask for a prediction. Then we package the whole thing into a **Docker image**, so it runs the same way on every machine.

### Step 0: 🛫 Before you start...
#### 🔀 Get to a feature branch
First, make sure that we are working on a feature branch.
```bash
git checkout -b feature/api
```

#### Get the model file
The API needs `wine_quality_model.pkl` in the **root** of the repository. This is the file that `download_model.py` created in Phase 4. If it is missing (e.g. fresh clone), download it again (the three `MLFLOW_*` environment variables from Phase 3 must be set in your terminal):
```bash
uv run download_model.py
```

#### 🔍 Check which features your model expects
The assignment says: *"Only include the features you trained on."* Don't guess – **ask the model**. A scikit-learn model that was trained on a `DataFrame` remembers its column names:
```bash
uv run python -c "import pickle; print(pickle.load(open('wine_quality_model.pkl', 'rb')).feature_names_in_)"
```

For the registered `rf_core_features` model you should see only the core 5 features.

> ⚠️ If you registered a different model in Phase 4 (e.g. `rf_baseline`, which uses all 11 features), use **the list you see here** in all following steps. The API must offer **exactly** the features of the registered model – not more, not fewer.

#### Install the two libraries we need
```bash
uv add fastapi uvicorn
```

- **`fastapi`** – the web framework: it defines the routes (`/predict`), reads and validates the request, and generates the interactive documentation for us. **Pydantic** is installed together with it automatically (FastAPI is built on top of it).
- **`uvicorn`** – the web *server*. FastAPI alone cannot listen on a port – uvicorn opens the port, speaks HTTP and hands every request over to our FastAPI app.

### Step 1: Create the file and the imports
Create `wine_quality_api.py` in the root of your project and start with the imports:

```python
import logging
import os
import pickle
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
```

What is new here?
- **`asynccontextmanager`** – needed for the *lifespan* function in Step 3 (code that runs at startup and shutdown).
- **`FastAPI`** – the application object.
- **`BaseModel`, `Field`, `ConfigDict`** – Pydantic building blocks to describe the structure of our JSON.

### Step 2: Define the configuration as constants
```python
MODEL_FILE = "wine_quality_model.pkl"

# Same features, same order as CORE_FEATURES in wine_quality_training.py
FEATURES = ["alcohol", "volatile_acidity", "sulphates", "total_sulfur_dioxide", "chlorides"]
```

**Why an explicit `FEATURES` list again?** <br> It is the "contract" of the model – the same one we wrote down in the training script (Phase 2/3). The API must hand the model **exactly these columns in exactly this order**. We use the list in Step 5 to build the input table.

### Step 3: Load the model once at startup
```python
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(f"{MODEL_FILE} not found. Run 'uv run download_model.py' first.")
    logger.info(f"Loading model from {MODEL_FILE}")
    with open(MODEL_FILE, "rb") as f:
        ml_models["wine_quality"] = pickle.load(f)
    logger.info("Model loaded")
    yield
    ml_models.clear()
```

What happens here, step by step:
- **`ml_models = {}`** – an (at first empty) dictionary at module level. It is the "shelf" where the loaded model lives while the API is running.
- **`lifespan`** – FastAPI runs everything **before the `yield`** once **when the server starts**, and everything **after the `yield`** once **when it shuts down**. That is exactly "load at startup".
- **The guard (`if not os.path.exists(...)`)** – the *fail fast* principle again: without a model file the server refuses to start and tells you what to do, instead of starting fine and crashing on the first request.
- **`"rb"`** = *read binary*, the counterpart of `"wb"` from Phase 2.
- **`ml_models.clear()`** – cleanup on shutdown.

🤔 **Why not simply load the model inside the `/predict` function?** <br>
Loading a forest with 200 trees from disk takes a moment. If we did it on every request, every single prediction would be slow. Loading **once at startup** and reusing the object makes each prediction a matter of milliseconds.

> ⚠️ **Security note:** `pickle.load` can execute arbitrary code that is hidden in a file. Only ever load pickle files you created yourself or fully trust – ours comes from *our own* model registry.

### Step 4: Define input and output with Pydantic
```python
class WineFeatures(BaseModel):
    alcohol: float = Field(ge=0, description="Alcohol content in % vol")
    volatile_acidity: float = Field(ge=0, description="Acetic acid in g/dm3")
    sulphates: float = Field(ge=0, description="Potassium sulphate in g/dm3")
    total_sulfur_dioxide: float = Field(ge=0, description="Total SO2 in mg/dm3")
    chlorides: float = Field(ge=0, description="Sodium chloride in g/dm3")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "alcohol": 10.5,
                    "volatile_acidity": 0.35,
                    "sulphates": 0.55,
                    "total_sulfur_dioxide": 115.0,
                    "chlorides": 0.05,
                }
            ]
        },
    )


class PredictionResponse(BaseModel):
    quality: float = Field(description="Predicted wine quality score")
```

What happens here:

- **A Pydantic model = a class that describes the shape of a JSON object.** Each attribute is a field with a type. `WineFeatures` is our **input**, `PredictionResponse` is our **output**.
- **Only 5 fields** – only the features the model was trained on. We don't ask the caller for `density`, `pH`, ... because the model could not use them anyway.
- **`Field(ge=0, ...)`** – `ge` = *greater or equal*. Negative alcohol content makes no sense, so it is rejected. `description` shows up in the documentation.
- **No default value** → the field is **required**. A request without `alcohol` is rejected.
- **`extra="forbid"`** – unknown fields (e.g. `"density": 0.99`) are rejected instead of being silently ignored. This enforces the "contract" strictly. (By default Pydantic would just ignore them.)
- **`json_schema_extra` / `examples`** – only cosmetic: the Swagger UI (Step 6) will pre-fill the request body with this example, so you can click "Execute" right away.

💡 **What does Pydantic do for us?** FastAPI hands every incoming JSON through the Pydantic model **before** our function runs: it (1) checks that all fields exist, (2) checks the types and constraints (and converts where reasonable, e.g. `"10.5"` → `10.5`), and (3) answers with an automatic **`422 Unprocessable Entity`** and a precise error message if something is wrong. Our function only ever sees clean data – we don't write a single `if` for this.

#### ❗ Don't forget to commit your changes!
```bash
git add wine_quality_api.py pyproject.toml uv.lock
git commit -m "Added model loading and input/output schemas for the API"
```

### Step 5: Create the app and the `/predict` endpoint
```python
app = FastAPI(title="Wine Quality API", version="0.1.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(wine: WineFeatures) -> PredictionResponse:
    X = pd.DataFrame([wine.model_dump()], columns=FEATURES)
    prediction = ml_models["wine_quality"].predict(X)[0]
    return PredictionResponse(quality=round(float(prediction), 2))
```

What happens here, line by line:

- **`app = FastAPI(..., lifespan=lifespan)`** – creates the application and plugs in our startup/shutdown function from Step 3. `title` and `version` appear in the generated docs.
- **`@app.post("/predict")`** – a *decorator* that registers the function below as the handler for **`POST /predict`**. (POST, because the caller *sends* data to us.)
- **`wine: WineFeatures`** – this single parameter is the magic: FastAPI sees the type, reads the JSON body, validates it with Pydantic (Step 4) and passes us a ready `WineFeatures` object.
- **`wine.model_dump()`** – turns the validated object into a plain dictionary.

🤔 **Why `def` and not `async def`?** <br> `model.predict` is CPU-bound, blocking code. FastAPI runs normal `def` endpoints in a **thread pool**, so a prediction doesn't block the server from handling other requests. With `async def` it *would* block the event loop.

### Step 6: Run the API locally
```bash
uv run uvicorn wine_quality_api:app --reload
```

What the command means:

- **`wine_quality_api:app`** – `<python file without .py>:<variable name of the FastAPI object>`. uvicorn imports `wine_quality_api.py` and looks for the variable `app`.
- **`--reload`** – restarts the server automatically when you save the file. Great for development, **not** for production.

You should see something like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
... INFO Loading model from wine_quality_model.pkl
... INFO Model loaded
INFO:     Application startup complete.
```

#### Try it in the auto-generated UI
1. Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** in your browser. This is the **Swagger UI** – FastAPI generated it from our code (routes, Pydantic models, descriptions).
2. Click **`POST /predict`** → **"Try it out"**. The request body is already filled with our example.
3. Click **"Execute"**. Under *Server response* you should see `200` and a body like:
   ```json
   { "quality": 5.97 }
   ```
   (The exact number depends on your model.)

#### Break it on purpose 🔨
Now let's see the Pydantic validation in action. Change the body in the UI and click **Execute** again:

| What you send | Result |
|---|---|
| remove the `alcohol` line | `422` – `"Field required"` |
| `"alcohol": "strong"` | `422` – value is not a valid number |
| `"alcohol": -1` | `422` – must be greater than or equal to 0 |
| add `"density": 0.99` | `422` – extra inputs are not permitted |

A `422` body looks like this (shortened):
```json
{ "detail": [ { "type": "missing", "loc": ["body", "alcohol"], "msg": "Field required" } ] }
```
`loc` tells you exactly *where* the problem is. **We didn't write any of this error handling** – Pydantic and FastAPI did.

#### The same thing with `curl`
In a second terminal:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol": 10.5, "volatile_acidity": 0.35, "sulphates": 0.55, "total_sulfur_dioxide": 115.0, "chlorides": 0.05}'
```

💡 **More things FastAPI generated for free:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) (alternative documentation) and [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json) (the machine-readable **OpenAPI** description that both UIs are built from).

Stop the server with `Ctrl+C`.

✅ **The first half of the task is done:** the API runs locally and predictions work via the UI.

#### ❗ Don't forget to commit your changes!
```bash
git add wine_quality_api.py
git commit -m "Added predict endpoint to the wine quality API"
```

### Step 7: Package the API as a Docker image (multi-stage)
Now we apply what we learned about Docker: a **multi-stage build** – one stage to **build** (install the dependencies), one **clean** stage to **run**.

#### 7.1 Create the `.dockerignore`
Docker sends the whole folder (the *build context*) to the Docker engine when you build. We don't want the virtual environment, the Git history or the DVC data in there. Create `.dockerignore` in the root:

```
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/

# Git & DVC (.dvc/config.local contains your DagsHub token!)
.git/
.gitignore
.github/
.dvc/
.dvcignore
data/

# ML artifacts we don't need in the API image
models/
mlruns/
notebooks/

# Scripts and docs that are not needed to serve predictions
wine_quality_training.py
download_model.py
ACTIONS.md
README.md

# IDE & OS
.idea/
.vscode/
.DS_Store
Thumbs.db

# Docker
.dockerignore
Dockerfile*
```

- 🔒 **Security:** In Phase 1 you stored your DagsHub token in `.dvc/config.local`. That file is ignored by *Git*, but Docker doesn't know that – without `.dockerignore` the token could end up inside the image.
- ⚡ **Speed:** `.venv/` and `data/` can be huge; excluding them makes the build context small.
- ⚠️ Do **not** ignore `wine_quality_model.pkl`, `wine_quality_api.py`, `pyproject.toml` or `uv.lock` – we need them!


#### 7.2 Write the `Dockerfile`
Create a file named `Dockerfile` (no extension) in the root. We build it piece by piece.

**Build argument**
```dockerfile
ARG PYTHON_VERSION=3.13
```
A variable for the Python version, so we change it in one place. It must match `requires-python` in `pyproject.toml` (`>=3.13`).

**The builder stage**
```dockerfile
# Builder stage: install dependencies
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# Never let uv download its own Python, always use the one of the image
ENV UV_PYTHON_DOWNLOADS=never

# Install curl and uv
RUN apt-get update && apt-get install -y curl
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy only the files that define the dependencies (-> layer caching)
COPY pyproject.toml uv.lock ./

# Install exactly the locked versions into /app/.venv
RUN uv sync --frozen --no-dev
```

- **`AS builder`** – names the stage, so we can copy from it later. `-slim` means it's stripped down to save space. It's like buying an unfurnished apartment.
- **`ENV UV_PYTHON_DOWNLOADS=never`** – the virtual environment (`.venv`) contains a *link* to the Python it was created with. If `uv` downloaded its own Python into the builder, that link would point to a place that **doesn't exist in the runtime stage** and the container would crash. With `never`, uv must use the Python of the base image (and stops with an error if there is none that fits).
- **`curl` + `uv` install** – exactly as in the lecture. We need `curl` only to download `uv`; it stays in the builder and **never reaches the final image**.
- **`COPY pyproject.toml uv.lock ./`** – only the dependency files, not the code. Docker caches each instruction as a **layer**: as long as these two files don't change, Docker re-uses the (slow) dependency layer when you only edit `wine_quality_api.py`.
- **`uv sync --frozen --no-dev`** – installs exactly the versions from `uv.lock` (`--frozen` = don't re-resolve) and skips development-only packages. 🔑 **This matters for our model:** the pickle was created with the scikit-learn version from your lock file. The image installs the very same version, so loading the model works. (A different scikit-learn version can break or warn when unpickling.)

> ❗ `uv.lock` must be committed to Git and exist locally – otherwise `--frozen` fails.

**The runtime stage**
```dockerfile
# Runtime stage: clean image with only what we need to run the API
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Take the installed environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Take only the code and the model
COPY wine_quality_api.py wine_quality_model.pkl ./

EXPOSE 8000

ENTRYPOINT ["/app/.venv/bin/uvicorn"]
CMD ["wine_quality_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- **`FROM ...` (second time)** – starts a **brand-new image**. Everything from the builder (curl, uv, caches) is left behind.
- **`COPY --from=builder /app/.venv /app/.venv`** – the magic line: only the finished environment moves over. It has to land at the **same path (`/app/.venv`)** because the scripts inside it (like `uvicorn`) contain the absolute path of the Python they belong to. Same `WORKDIR` + same Python base image in both stages = it works.
- **`COPY wine_quality_api.py wine_quality_model.pkl ./`** – we are selective: the API code and the model. No training script, no data, no Git history.
- **`EXPOSE 8000`** – *documentation only*: "this container listens on port 8000". It does **not** open the port – that is `-p` at `docker run`.
- **`ENTRYPOINT` + `CMD`** – as in the `hello-world` example: `ENTRYPOINT` is the program (`uvicorn`), `CMD` are its default arguments (which could be overwritten at `docker run`).
- 🔑 **`--host 0.0.0.0`** – by default uvicorn listens on `127.0.0.1` (*localhost*). Inside a container that means "reachable only from inside this very container" – port mapping couldn't reach it. `0.0.0.0` means "listen on all network interfaces".

🤔 **Why is the model copied *into* the image?** <br> The image becomes **self-contained and immutable**: image + tag = exactly one model version, and it runs anywhere without access to the registry or credentials. The price: to ship a new model you rebuild the image. Because the model comes from `download_model.py`, a CI pipeline would have to run that script *before* `docker build`.

#### 7.3 Build and run
```bash
docker build -t wine-quality-api .
```
The `-t` tag names the image as `wine-quality-api`, the `.` is the build context (the current folder).

If you see `COPY failed: file not found ... wine_quality_model.pkl`, run `uv run download_model.py` first (Step 0).

You may also get `ERROR: permission denied ...`, check whether docker is active with `sudo systemctl status docker`.

Now start a container (make sure your local `uvicorn` from Step 6 is stopped, otherwise port 8000 is taken):
```bash
docker run --rm -p 8000:8000 wine-quality-api
```

- **`--rm`** – remove the container when it stops (lecture best practice).
- **`-p 8000:8000`** – `<port on your machine>:<port in the container>`. Port 8000 on your laptop is forwarded to port 8000 in the container. (Port busy? Use `-p 8080:8000` and open `localhost:8080`.)

Open **[http://localhost:8000/docs](http://localhost:8000/docs)** and run the same prediction as in Step 6 – this time it is answered **by the container**. 🎉 Stop it with `Ctrl+C`.

#### 7.4 Inspect the result
How lean is our image, and is it really clean?
```bash
docker images wine-quality-api
docker run -it --rm --entrypoint sh wine-quality-api
```
Inside the container:
```bash
ls -la
which curl
exit
```
You should see only `.venv`, `wine_quality_api.py` and `wine_quality_model.pkl` – and `which curl` finds **nothing**. The build tools stayed in the builder stage. That is the point of a multi-stage build: smaller image, smaller attack surface, only runtime content.

✅ **The second half is done:** the Docker image builds and the API works inside the container.

#### ❗ Let's commit and wrap up the phase!
```bash
git add wine_quality_api.py Dockerfile .dockerignore pyproject.toml uv.lock
git commit -m "Wrapped up everything! API served in a multi-stage Docker image"
git switch main
git merge feature/api
git push origin main
git branch -d feature/api
```

🎉 This section is finished! Our model is now a real service: validated input, one-time model loading, auto-generated documentation – and packaged so that it runs anywhere Docker runs.


## ⚙️ Phase 6: GitHub & CI Pipeline
> Configure it to be a safe working environment
> 
> - Protect the trunk. Make sure that nobody can push to it or merge any PR without checks passing
> - Set up branch policies to require PR reviews and passing checks
> - Do not allow secrets to be added to Git
> 
> Create a GitHub Actions workflow for training:
> 
> - Triggers if training relevant data or code has changed. Allows manual dispatch
> - Pulls the data with DVC
> - Runs `wine_quality_training.py` and logs results to MLflow
> 
> Create a GitHub Actions workflow for the API:
> 
> - Builds the Docker image and pushes it to the GitHub Container Registry (GHCR)
> 
> Make sure that both workflows 
> 
> - Run linting on the code
> - Run secret scanning and security checks
> - Run the unit tests
> 
> ✅ Every push to `main` triggers both workflows; the Docker image is available on GHCR.

In this phase, we transition from running everything manually on our computer to automated **Continuous Integration (CI)** with GitHub Actions.

The important idea is that `main` becomes our **trunk**: we don't just hope that the code is correct — GitHub actively prevents broken code from being merged. Every relevant change is checked automatically, training can be reproduced on a clean GitHub runner, and the API is packaged and published automatically.

### Step 0: 🛫 Switch to a feature branch & set up dev tools

Start by creating a feature branch. We don't want to experiment with branch protection and workflows directly on `main`.

~~~bash
git checkout -b feature/ci-pipeline
~~~

To support automated linting and unit testing, add the development dependencies:

~~~bash
uv add --dev ruff pytest httpx2
~~~

Now synchronize the lock file:

~~~bash
uv sync
~~~

> 💡 `ruff` is our linter. `pytest` is our unit testing framework. They are development dependencies because they are needed to **develop and verify** the project, but they are not needed to run the production API.

#### Update `pyproject.toml`
In order to run tests, we have to update our `pyproject.toml` file. Add this to the end of it:
~~~toml
[tool.pytest.ini_options]
pythonpath = ["."]
~~~

### Step 1: Add a unit test

We need an actual test suite before we can require tests to pass in CI.

Create the directory and file:

~~~bash
mkdir -p tests
touch tests/test_api.py
~~~

Put this into `tests/test_api.py`:

~~~python
from fastapi.testclient import TestClient

from wine_quality_api import app


client = TestClient(app)


def test_docs_endpoint():
    response = client.get("/docs")

    assert response.status_code == 200
~~~

Run it locally:

~~~bash
uv run pytest
~~~

You should see that one test passed.

#### ❗ Don't forget to commit your changes!
As everything worked out, save this properly with git.
~~~bash
git add tests/ pyproject.toml uv.lock
git commit -m "Added first unit test"
~~~

### Step 2: 🔍 Check the code locally before CI
Before making GitHub do the work for us, run the same checks locally.

#### Lint the project
~~~bash
uv run ruff check .
~~~

If everything is fine, Ruff should report:

~~~text
All checks passed!
~~~

If it finds problems, fix them before continuing. You can do this automatically with:
~~~bash
uv run ruff format .
~~~

If more problems occur, try also `uv run ruff check . --fix`.

> 🤔 **Linting vs formatting:** linting looks for problematic code, while formatting changes the appearance of the code to follow a consistent style. We will use `ruff check` as the CI gate because it can fail the workflow when it finds a problem.

#### Run the tests

~~~bash
uv run pytest
~~~

#### Run the dependency security scan

The course CI example uses `uv-secure` to inspect the versions recorded in `uv.lock`:

~~~bash
uvx uv-secure --config=pyproject.toml
~~~

This checks the project's dependencies against known vulnerabilities.

> ⚠️ A security scanner does not prove that the application is secure. It is one automated layer of defence. It is especially useful for catching vulnerable third-party dependencies.

#### ❗ Don't forget to commit your changes!
We reformatted all of your python files, we want to save these changes.
~~~bash
git add wine_quality_training.py wine_quality_api.py download_model.py
git commit -m "Ran ruff checks & reformatting"
~~~

### Step 3: 🛡️ Secure the GitHub repository

Before creating the workflows, configure GitHub itself so that the CI checks actually matter.

#### 3.1 Enable Secret Scanning and Push Protection

On GitHub:

1. Open your repository.
2. Go to **Settings** → **Code security and analysis**.
3. Enable **Secret scanning** if it is available.
4. Enable **Push protection** if it is available.

> 🔒 **What does Push Protection do?**
>
> Suppose you accidentally write:
>
> ~~~python
> DAGS_HUB_TOKEN = "my-real-token"
> ~~~
>
> and try to push it. Push protection can detect the credential-like value and block the push before the secret reaches the remote repository.
>
> This is especially important for this project because DagsHub/MLflow credentials are needed by CI. **Credentials belong in GitHub Secrets, never in Python files, YAML files, `.dvc/config`, or committed configuration.**

#### 3.2 Add the CI secrets

Our GitHub Actions runners need credentials to access the DagsHub DVC remote and MLflow server.

Go to:

**Settings → Secrets and variables → Actions → New repository secret**

Create:

| Secret | What it contains |
|---|---|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | Your DagsHub access token |

We will use these secrets for both DVC and MLflow.

> 🔐 **Why two GitHub Secrets instead of writing the values directly into the workflow?**
>
> GitHub Secrets are encrypted and exposed to the workflow only when explicitly referenced. The YAML file can therefore be committed safely.
>
> Never do this:
>
> ~~~yaml
> MLFLOW_TRACKING_PASSWORD: "my-real-password"
> ~~~
>
> Instead:
>
> ~~~yaml
> MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
> ~~~

### Step 4: Protect the `main` branch
Now we make the trunk safe. Go to **Settings → Rules → Rulesets → New branch ruleset**

Configure the rule/ruleset for: `main`

Enable the following:

1. **Require a pull request before merging**
  - Require at least **1 approval**
2. **Require status checks to pass before merging**
3. **Require branches to be up to date before merging**
4. **Do not allow bypassing the above settings**

Also make sure that direct pushes to `main` are not permitted.

After the workflows have run at least once, GitHub will know their check names. Select the checks corresponding to our CI jobs:

~~~text
Lint and Test
Security Checks
Train and Track
Build and Push GHCR
~~~

> ⚠️ **Important:** GitHub can only require a status check that has actually appeared in the repository. If you create the branch protection rule before the workflows have run, the required-check list may not contain the names yet. Run the workflows once, then return to the branch protection settings.

### Step 5: Create the training workflow

GitHub Actions looks for workflow files in:

~~~text
.github/workflows/
~~~

Create the directory and workflow:

~~~bash
mkdir -p .github/workflows
touch .github/workflows/training.yml
~~~

The training workflow needs to:

1. Run on relevant pull requests.
2. Run on every push to `main`.
3. Allow manual execution.
4. Install the project.
5. Run Ruff.
6. Run the security scan.
7. Run the unit tests.
8. Configure DVC credentials.
9. Pull the Parquet dataset with DVC.
10. Configure MLflow.
11. Run `wine_quality_training.py`.

Put the following into `.github/workflows/training.yml`:

~~~yaml
name: Training Pipeline

on:
  push:
    branches:
      - main

  pull_request:
    branches:
      - main
    paths:
      - "wine_quality_training.py"
      - "data/**"
      - "*.dvc"
      - "data/*.dvc"
      - "dvc.yaml"
      - "dvc.lock"
      - "pyproject.toml"
      - "uv.lock"
      - ".github/workflows/training.yml"

  workflow_dispatch:

permissions:
  contents: read

jobs:
  lint-and-test:
    name: Lint and Test
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Install project
        run: uv sync --frozen

      - name: Lint code
        run: uv run ruff check . --output-format=github

      - name: Run unit tests
        run: uv run pytest

  security:
    name: Security Checks
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Scan dependencies
        run: |
          set -o pipefail
          echo "### 🛡️ Security Audit Results" >> "$GITHUB_STEP_SUMMARY"
          echo '```text' >> "$GITHUB_STEP_SUMMARY"
          uvx uv-secure --config=pyproject.toml | tee -a "$GITHUB_STEP_SUMMARY"
          echo '```'

      - name: Scan for secrets
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  train:
    name: Train and Track
    runs-on: ubuntu-latest
    needs:
      - lint-and-test
      - security

    env:
      DAGSHUB_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
      DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
      MLFLOW_TRACKING_URI: https://dagshub.com/${{ secrets.DAGSHUB_USERNAME }}/ais-devil2-wine-quality-full.mlflow
      MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Install project
        run: uv sync --frozen

      - name: Configure DVC credentials
        run: |
          uv run dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
          uv run dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"

      - name: Pull training data
        run: uv run dvc pull

      - name: Train model and log to MLflow
        run: uv run wine_quality_training.py rf_baseline
~~~

#### ❗ Don't forget to commit your changes!
Let's commit now the created `training.yml` file!
~~~bash
git add .github/workflows/training.yml
git commit -m "Added training workflow"
~~~

### Step 6: Understand the training workflow
There are several important ideas here that are worth noting.

#### `on.push`

~~~yaml
push:
  branches:
    - main
~~~

This makes every push to `main` start the training workflow.

This is important because the final assignment explicitly requires that **every push to `main` triggers both workflows**.

#### `on.pull_request.paths`

~~~yaml
pull_request:
  branches:
    - main
  paths:
    - "wine_quality_training.py"
    - "data/**"
    - "*.dvc"
    - "pyproject.toml"
    - "uv.lock"
~~~

This prevents irrelevant pull requests from retraining the model.

For example, changing only documentation should not require a new training run.

A change to the training script (`wine_quality_training.py`), as well as a change to the data or the environment (`pyproject.toml` and `uv.lock` files).

> 💡 This is a useful concept: **path filters reduce unnecessary CI work**, while the unconditional `push` to `main` satisfies the assignment's requirement that every successful merge/push to the trunk starts both pipelines.

#### `workflow_dispatch`

~~~yaml
workflow_dispatch:
~~~

This adds the **Run workflow** button to GitHub Actions. It is useful when you want to retrain manually without changing any code,for example, if you want to verify that the training pipeline still works after changing something on DagsHub, you can start it manually.

#### `needs`

~~~yaml
needs:
  - lint-and-test
  - security
~~~

This means the `train` job waits until both prerequisite jobs succeed.
If linting or security scanning fails, training does not start.

#### `uv sync --frozen`
This is particularly important in CI.
- `uv.lock` records the exact dependency versions.
- `--frozen` tells `uv`:

This makes the CI environment reproducible.

### Step 7: Configure DVC inside CI
Your local machine already has DVC configured, but the GitHub runner starts from a clean environment. That means it does **not** have your local `.dvc/config.local`. We therefore configure the credentials during the workflow:

~~~bash
uv run dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
uv run dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"
~~~

The important word here is **`--local`**: tt writes the credentials to DVC's local configuration rather than committing them to the repository.

Remember, git only contains a pointer to the actual data and that is stored on Dagshub with Data Version Control. This is what we did in Phase 1.

### Step 8: Configure MLflow inside CI

The training script already contains this fail-fast check:

~~~python
REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
]
~~~

and:

~~~python
def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
~~~

Therefore the workflow simply provides these variables through its `env` section:

~~~yaml
env:
  MLFLOW_TRACKING_URI: https://dagshub.com/${{ secrets.DAGSHUB_USERNAME }}/ais-devil2-wine-quality-full.mlflow
  MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
  MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
~~~

The training script does not need to know whether it is running on your laptop or on GitHub Actions.

> 💡 **That is an important CI principle:** the code stays the same; the environment provides the configuration.

### Step 9: Create the API workflow

Now create the second workflow:

~~~bash
touch .github/workflows/api.yml
~~~

This workflow needs to:

1. Run on every push to `main`.
2. Run on pull requests.
3. Allow manual execution.
4. Run linting.
5. Run security checks.
6. Run unit tests.
7. Download the registered model.
8. Build the Docker image.
9. Authenticate with GHCR.
10. Push the image to GHCR.

Put this into `.github/workflows/api.yml`:

~~~yaml
name: API Pipeline

on:
  push:
    branches:
      - main

  pull_request:
    branches:
      - main

  workflow_dispatch:

permissions:
  contents: read

jobs:
  lint-and-test:
    name: Lint and Test
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Install project
        run: uv sync --frozen

      - name: Lint code
        run: uv run ruff check . --output-format=github

      - name: Run unit tests
        run: uv run pytest

  security:
    name: Security Checks
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Scan dependencies
        run: |
          set -o pipefail
          echo "### 🛡️ Security Audit Results" >> "$GITHUB_STEP_SUMMARY"
          echo '```text' >> "$GITHUB_STEP_SUMMARY"
          uvx uv-secure --config=pyproject.toml | tee -a "$GITHUB_STEP_SUMMARY"
          echo '```'

      - name: Scan for secrets
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  build-and-push:
    name: Build and Push GHCR
    runs-on: ubuntu-latest
    needs:
      - lint-and-test
      - security

    permissions:
      contents: read
      packages: write

    env:
      MLFLOW_TRACKING_URI: https://dagshub.com/${{ secrets.DAGSHUB_USERNAME }}/ais-devil2-wine-quality-full.mlflow
      MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Install project
        run: uv sync --frozen

      - name: Download registered model
        run: uv run download_model.py

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository_owner }}/wine-quality-api
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha

      - name: Build and push Docker image
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
~~~

### Step 10: 🤔 Understand the GHCR part

The GitHub Container Registry is GitHub's registry for container images.

The important line is:

~~~yaml
registry: ghcr.io
~~~

The image name is generated from:

~~~yaml
ghcr.io/${{ github.repository_owner }}/wine-quality-api
~~~

For this repository, the resulting image will look conceptually like:

~~~text
ghcr.io/pleventel/wine-quality-api
~~~

The exact capitalization/normalisation of the image name follows GHCR's requirements.

#### Why do we need `packages: write`?

The workflow initially has:

~~~yaml
permissions:
  contents: read
~~~

because we want the default token to have as little access as possible.

The build job then explicitly requests:

~~~yaml
permissions:
  contents: read
  packages: write
~~~

This gives that job permission to publish a package to GHCR.

This is the *principle of least privilege*: Give a workflow only the permissions it actually needs.

#### Why use `GITHUB_TOKEN` instead of another Docker password?

GitHub automatically creates:

~~~text
secrets.GITHUB_TOKEN
~~~

for each workflow run. 
It can authenticate the workflow against GitHub services, including GHCR. This means we don't need to create another personal GitHub access token just to publish the image.

### Step 11: Why does the API workflow download the model?
Remember the Dockerfile from Phase 5:

~~~dockerfile
COPY wine_quality_api.py wine_quality_model.pkl ./
~~~

The Docker image therefore **needs the model file to exist before `docker build` starts**.

But `.gitignore` deliberately keeps `wine_quality_model.pkl` out of Git, so a fresh GitHub runner does not have the model.

The workflow solves this by running:

~~~bash
uv run download_model.py
~~~

The model version in `.model-version` determines what gets downloaded, and that exact model becomes part of the Docker image.

### Step 12: Test the workflows
As we were continously commiting everything, there's nothing really new to commit, but notice, we haven't pushed our work yet. Let's do it now.
~~~bash
git push -u origin feature/ci-pipeline
~~~

Now open your repository on GitHub and go to:

**Actions**

You should see both workflows.

You can also create a Pull Request:

~~~text
feature/ci-pipeline → main
~~~

The PR should show the required checks.

### Step 13: Check the Docker image in GHCR
After `api.yml` completes successfully, open your GitHub repository.

Look for the **Packages** section.

You should find:

~~~text
wine-quality-api
~~~

The image should have at least a `latest` tag when the workflow ran from the default branch.

You can also pull it locally:

~~~bash
docker pull ghcr.io/<YOUR_GITHUB_USERNAME>/wine-quality-api:latest
~~~

Then run it:

~~~bash
docker run --rm -p 8000:8000 ghcr.io/<YOUR_GITHUB_USERNAME>/wine-quality-api:latest
~~~

Now open [http://localhost:8000/docs](http://localhost:8000/docs) and test the `/predict` endpoint.

🎉 The Docker image that you just ran was not built manually on your laptop — it was built by CI and published to GHCR.

### Step 14: Pull Request and merge
If you go to your github repository, you'll see that there're commits ready to merge. Create a pull request and merge them!

If everything was successful, you can now merge your branch and delete your feature branch too.
~~~bash
git merge feature/ci-pipeling
git push origin main
~~~

🎉 This section is also officially finished!

