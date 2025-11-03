import click, os, yaml
from conversation_vad_labeler.pipeline import run_pipeline

@click.command()
@click.option('--experiment', required=True, type=int)
@click.option('--trial', required=True, type=int)
@click.option('--config', default='config.yaml')
@click.option('--language', default=None)
@click.option('--device', default='cpu')
def cli(experiment, trial, config, language, device):
    cfg_path = config if os.path.exists(config) else None
    outputs = run_pipeline(experiment, trial, cfg_path=cfg_path, overwrite=False, device=device, language=language)
    print('Outputs:'); print(outputs)
if __name__=='__main__': cli()
 

 