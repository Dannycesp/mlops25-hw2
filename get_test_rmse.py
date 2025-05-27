import mlflow

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Get the experiment
experiment = mlflow.get_experiment_by_name("random-forest-best-models")

# Get all runs in the experiment
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Find the run with the lowest test RMSE
best_run = runs.loc[runs['metrics.test_rmse'].idxmin()]
best_test_rmse = best_run['metrics.test_rmse']

print(f"Best test RMSE: {best_test_rmse:.3f}")
