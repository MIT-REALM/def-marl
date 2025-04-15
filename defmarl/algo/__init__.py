from .base import Algorithm
from .informarl import InforMARL
from .defmarl import DefMARL
from .informarl_lagr import InforMARLLagr


def make_algo(algo: str, **kwargs) -> Algorithm:
    if algo == 'informarl':
        return InforMARL(**kwargs)
    elif algo == 'def-marl':
        return DefMARL(**kwargs)
    elif algo == 'informarl_lagr':
        return InforMARLLagr(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
