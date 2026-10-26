"""Restart preservation policy only; never restarts the running service."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('recovery_supervisor_checks',Path(__file__).with_name('resume_training_v2.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_finished_outcomes_inputs_and_versioned_models_remain_protected():
    for path in ('actors/nn/train/episode/complete.json','datasets/development_manifest.json',
                 'datasets/train/episode.json','models/nn/bootstrap.json','models/xgboost/version123.json'):
        assert module.immutable_preservation_path(module.ROOT/path)

def test_legitimate_checkpoint_and_live_pointer_progress_does_not_block_restart():
    for path in ('actors/nn/train/episode/checkpoint.json','actors/nn/train/episode/checkpoint.previous.json',
                 'actors/nn/train/episode/states/00050.pickle','models/nn/current.json',
                 'models/nn/last_fit_runtime.json'):
        assert not module.immutable_preservation_path(module.ROOT/path)
