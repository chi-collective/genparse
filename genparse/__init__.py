import importlib


def __getattr__(name):
    module_name, attr_name = _lazy_imports.get(name, (None, None))
    if module_name is None:
        raise AttributeError(f'module {__name__} has no attribute {name}')

    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr  # Cache the attribute in the globals for future access
    return attr


_lazy_imports = {
    'CFG': ('genparse.cfg', 'CFG'),
    'Derivation': ('genparse.cfg', 'Derivation'),
    'Rule': ('genparse.cfg', 'Rule'),
    'prefix_transducer': ('genparse.cfg', 'prefix_transducer'),
    'EOS': ('genparse.cfglm', 'EOS'),
    'add_EOS': ('genparse.cfglm', 'add_EOS'),
    'locally_normalize': ('genparse.cfglm', 'locally_normalize'),
    'BoolCFGLM': ('genparse.cfglm', 'BoolCFGLM'),
    'Chart': ('genparse.chart', 'Chart'),
    'FST': ('genparse.fst', 'FST'),
    'MockLLM': ('genparse.lm', 'MockLLM'),
    'LM': ('genparse.lm', 'LM'),
    'LLM': ('genparse.lm', 'LLM'),
    'TokenizedLLM': ('genparse.lm', 'TokenizedLLM'),
    'Boolean': ('genparse.semiring', 'Boolean'),
    'Entropy': ('genparse.semiring', 'Entropy'),
    'Float': ('genparse.semiring', 'Float'),
    'Log': ('genparse.semiring', 'Log'),
    'MaxPlus': ('genparse.semiring', 'MaxPlus'),
    'MaxTimes': ('genparse.semiring', 'MaxTimes'),
    'Real': ('genparse.semiring', 'Real'),
    'EPSILON': ('genparse.wfsa', 'EPSILON'),
    'WFSA': ('genparse.wfsa', 'WFSA'),
    'load_model_by_name': ('genparse.util', 'load_model_by_name'),
    'lark_guide': ('genparse.util', 'lark_guide'),
    'InferenceSetup': ('genparse.util', 'InferenceSetup'),
    'EarleyLM': ('genparse.parse.earley', 'EarleyLM'),
    'Earley': ('genparse.parse.earley', 'Earley'),
    'InferenceSetupVLLM': ('genparse.backends.vllm', 'InferenceSetupVLLM'),
}

__all__ = list(_lazy_imports.keys())
