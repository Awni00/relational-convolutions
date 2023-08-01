from tqdm import tqdm, trange
import wandb

def evaluate_learning_curves(
    create_model, eval_model, fit_kwargs, create_callbacks,
    wandb_project_name, group_name,
    train_ds, val_ds, test_ds,
    train_sizes, start_trial, num_trials,
    ):

    for train_size in tqdm(train_sizes, desc='train size'):

        for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
            run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial, 'group': group_name})
            model = create_model()

            train_ds_sample = train_ds.shuffle(buffer_size=len(train_ds)).take(train_size).batch(batch_size)
            history = model.fit(
                train_ds_sample, validation_data=val_ds, verbose=0,
                callbacks=create_callbacks(), **fit_kwargs)

            eval_dict = eval_model(model)
            wandb.log(eval_dict)
            wandb.finish(quiet=True)

            del model, train_ds_sample