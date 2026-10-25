"""The pilot cannot silently expand into a full or held-out-city campaign."""
import runpy
from pathlib import Path

import pytest

from fleet_switching.config import DEVELOPMENT, HELD_OUT, atomic


def limits():
    return {'development_cities':sorted(DEVELOPMENT),'held_out_cities':sorted(HELD_OUT),
        'training_inputs_total':16,'training_actor_episodes_total':32,'worker_limit':8,
        'maximum_collection_wall_seconds':7200,'full_training_authorized':False}


@pytest.mark.parametrize('change',[
    {'training_inputs_total':800},
    {'worker_limit':64},
    {'full_training_authorized':True},
    {'development_cities':sorted(DEVELOPMENT|{'ghent'})},
])
def test_pilot_scope_rejects_expansion_or_test_cities(tmp_path,change):
    namespace=runpy.run_path(str(Path(__file__).parents[1]/'operations/switching_pilot.py'))
    value=limits();value.update(change)
    atomic(tmp_path/'PILOT_LIMITS.json',value)
    with pytest.raises(ValueError,match='Pilot scope changed'):
        namespace['guard_scope'](tmp_path)
