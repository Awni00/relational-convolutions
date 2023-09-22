import wandb
import pandas as pd
from tqdm import tqdm

def get_wandb_project_table(project_name, entity='Awni00', attr_cols=('group', 'name'), config_cols='all', summary_cols='all'):

    api = wandb.Api()

    runs = api.runs(entity + "/" + project_name)

    if summary_cols == 'all':
        summary_cols = set().union(*tuple(run.summary.keys() for run in runs))

    if config_cols == 'all':
        config_cols = set().union(*tuple(run.config.keys() for run in runs))

    all_cols = list(attr_cols) + list(summary_cols) + list(config_cols)
    if len(all_cols) > len(set(all_cols)):
        raise ValueError("There is overlap in the `config_cols`, `attr_cols`, and `summary_cols`")

    data = {key: [] for key in all_cols}

    for run in runs:
        for summary_col in summary_cols:
            data[summary_col].append(run.summary.get(summary_col, None))

        for config_col in config_cols:
            data[config_col].append(run.config.get(config_col, None))

        for attr_col in attr_cols:
            data[attr_col].append(getattr(run, attr_col, None))

    runs_df = pd.DataFrame(data)

    return runs_df

def get_project_run_histories(project_name, entity='Awni00', attr_cols=('group', 'name'), config_cols='all'):
    '''gets the log history of all runs in a project'''

    def get_run_history(run):
        history_scan = run.scan_history()
        try:
            keys = history_scan.next().keys()
        except StopIteration:
            print(f'{run.group}-{run.name} has no history. Skipping...')
            return pd.DataFrame()
        run_history_data = {key: [] for key in keys}
        for row in history_scan:
            for key in keys:
                run_history_data[key].append(row[key])
        return pd.DataFrame(run_history_data)

    api = wandb.Api(timeout=60)

    runs = api.runs(entity + "/" + project_name)

    if config_cols == 'all':
        config_cols = set().union(*tuple(run.config.keys() for run in runs))

    run_history_dfs = []

    for run in tqdm(runs, leave=False):
        run_history = get_run_history(run)

        for config_col in config_cols:
            run_history[config_col] = run.config.get(config_col, None)

        for attr_col in attr_cols:
            run_history[attr_col] = getattr(run, attr_col, None)

        run_history_dfs.append(run_history)

    runs_history_df = pd.concat(run_history_dfs, axis=0)

    runs_history_df = runs_history_df.reset_index(drop=True)

    return runs_history_df