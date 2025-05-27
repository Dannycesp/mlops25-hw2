import mlflow

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Get the experiment
experiment = mlflow.get_experiment_by_name("random-forest-hyperopt")

# Get all runs in the experiment
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Find the run with the lowest RMSE
best_run = runs.loc[runs['metrics.rmse'].idxmin()]
best_rmse = best_run['metrics.rmse']

print(f"Best RMSE: {best_rmse:.3f}")
