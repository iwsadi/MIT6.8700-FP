import argparse
import json
import importlib
import torch
from dacite import from_dict

from smiles_rl.configuration_envelope import ConfigurationEnvelope
from smiles_rl.utils.general import _set_torch_device


def load_dynamic_class(name_spec: str):
    if name_spec is None:
        raise KeyError("Key method not found in config")
    if "." not in name_spec:
        raise ValueError("Must provide full module path in name_spec")
    module_name, name = name_spec.rsplit(".", maxsplit=1)
    loaded_module = importlib.import_module(module_name)
    if not hasattr(loaded_module, name):
        raise ValueError(f"Module ({module_name}) does not have a class called {name}")
    return getattr(loaded_module, name)


def _read_json_file(path: str):
    with open(path) as f:
        raw = f.read().replace("\r", "").replace("\n", "")
    return json.loads(raw)


def _construct_logger(config: ConfigurationEnvelope):
    method_class = load_dynamic_class(config.logging.method)
    return method_class(config)


def _construct_scoring_function(config: ConfigurationEnvelope):
    method_class = load_dynamic_class(config.scoring_function.method)
    return method_class(config)


def _construct_diversity_filter(config: ConfigurationEnvelope):
    method_class = load_dynamic_class(config.diversity_filter.method)
    return method_class(config)


def _construct_replay_buffer(config: ConfigurationEnvelope):
    method_class = load_dynamic_class(config.replay_buffer.method)
    # Replay buffers expect the parameters dict, not the full config
    return method_class(config.replay_buffer.parameters)


def _construct_agent(config: ConfigurationEnvelope, logger, scoring_function, diversity_filter, replay_buffer):
    method_class = load_dynamic_class(config.reinforcement_learning.method)
    agent = method_class(
        config,
        scoring_function,
        diversity_filter,
        replay_buffer,
        logger,
    )
    return agent


def _run_one_step(batch_size, agent):
    smiles = agent.act(batch_size)
    assert len(smiles) <= batch_size, "Generated more SMILES strings than requested"
    agent.update(smiles)


def run(config: ConfigurationEnvelope, agent):
    batch_size = config.reinforcement_learning.parameters.batch_size
    n_steps = config.reinforcement_learning.parameters.n_steps
    for _ in range(n_steps):
        _run_one_step(batch_size, agent)
    agent.log_out()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config.json")
    args = parser.parse_args()

    config_dict = _read_json_file(args.config)
    config = from_dict(ConfigurationEnvelope, config_dict)

    print("Setting torch device based on availability", flush=True)
    use_cuda_flag = True
    try:
        use_cuda_flag = config.system.use_cuda  # type: ignore[attr-defined]
    except Exception:
        use_cuda_flag = torch.cuda.is_available()
    device_str = "cuda" if (use_cuda_flag and torch.cuda.is_available()) else "cpu"
    _set_torch_device(device_str)

    logger = _construct_logger(config)
    scoring_function = _construct_scoring_function(config)
    diversity_filter = _construct_diversity_filter(config)
    replay_buffer = _construct_replay_buffer(config)
    agent = _construct_agent(config, logger, scoring_function, diversity_filter, replay_buffer)

    print("Starting Transformer RL run (no distillation in this script)", flush=True)
    run(config, agent)


if __name__ == "__main__":
    main()

