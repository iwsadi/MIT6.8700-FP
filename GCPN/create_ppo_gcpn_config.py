import os
import json
import argparse


def main():
    args = _get_arguments()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # If agent not provided, use prior
    agent_path = args.agent if args.agent else args.prior

    config_json = _create_json_config(
        prior_model_path=args.prior,
        agent_model_path=agent_path,
        output_dir=log_dir,
        predictive_model_path=args.predictive_model,
        replay_buffer=args.replay_buffer,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gcpn_hidden_dim=args.gcpn_hidden_dim,
        gcpn_max_gen_steps=args.gcpn_max_gen_steps,
        gcpn_gen_attempts=args.gcpn_gen_attempts,
    )

    save_path_config = os.path.join(log_dir, "ppo_gcpn_config.json")
    with open(save_path_config, "w") as f:
        json.dump(config_json, f, indent=4, sort_keys=True)

    print(save_path_config)


def _get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PPO-GCPN config")

    parser.add_argument(
        "--replay_buffer",
        required=True,
        type=str,
        help="Absolute/relative import path to replay buffer class (e.g., smiles_rl.replay_buffer.all_current.AllCurrent)",
    )
    parser.add_argument(
        "--log_dir",
        required=True,
        type=str,
        help="Logging directory for saving config file",
    )
    parser.add_argument(
        "--prior",
        required=True,
        type=str,
        help="Path to GCPN prior weights (state_dict) (used for critic backbone init)",
    )
    parser.add_argument(
        "--agent",
        required=False,
        type=str,
        default=None,
        help="Path to GCPN agent weights (state_dict) (used for actor init). Defaults to --prior",
    )
    parser.add_argument(
        "--predictive_model",
        required=True,
        type=str,
        help="Path to predictive model for scoring (e.g., predictive_models/DRD2/RF_DRD2_ecfp4c.pkl)",
    )
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # GCPN-specific knobs
    parser.add_argument("--gcpn_hidden_dim", type=int, default=64)
    parser.add_argument("--gcpn_max_gen_steps", type=int, default=80)
    parser.add_argument("--gcpn_gen_attempts", type=int, default=50)

    return parser.parse_args()


def _create_json_config(
    prior_model_path: str,
    agent_model_path: str,
    output_dir: str,
    predictive_model_path: str,
    replay_buffer: str,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    gcpn_hidden_dim: int,
    gcpn_max_gen_steps: int,
    gcpn_gen_attempts: int,
):
    results_dir = os.path.join(output_dir, "results")

    configuration = {}

    configuration["logging"] = {
        "method": "smiles_rl.logging.reinforcement_logger.ReinforcementLogger",
        "parameters": {
            "sender": "http://127.0.0.1",
            "recipient": "local",
            "logging_frequency": 0,
            "logging_path": os.path.join(output_dir, "progress_log"),
            "result_folder": results_dir,
            "job_name": "ppo gcpn",
            "job_id": "ppo_gcpn",
        },
    }

    configuration["diversity_filter"] = {
        "method": "smiles_rl.diversity_filter.diversity_filter_factory.DiversityFilterFactory",
        "parameters": {
            "name": "IdenticalMurckoScaffold",
            "bucket_size": 25,
            "minscore": 0.4,
            "minsimilarity": 0.35,
        },
    }

    configuration["reinforcement_learning"] = {
        "method": "smiles_rl.agent.ppo_gcpn.PPOGCPN",
        "parameters": {
            "prior": prior_model_path,
            "agent": agent_model_path,
            "n_steps": n_steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "specific_parameters": {
                "clip": 0.2,
                "n_updates_per_iteration": 4,
                "discount_factor": 0.99,
                "use_entropy_bonus": False,
                "entropy_coeff": 0.001,
                "max_grad_norm": 0.5,
                "n_minibatches": 4,
                # GCPN knobs
                "gcpn_hidden_dim": gcpn_hidden_dim,
                "gcpn_max_gen_steps": gcpn_max_gen_steps,
                "gcpn_gen_attempts": gcpn_gen_attempts,
            },
        },
    }

    configuration["replay_buffer"] = {
        "method": replay_buffer,
        "parameters": {
            "k": batch_size // 2,
            "memory_size": 1000,
        },
    }

    configuration["scoring_function"] = {
        "method": "smiles_rl.scoring.reinvent_scoring_factory.ReinventScoringFactory",
        "parameters": {
            "name": "custom_sum",
            "parallel": False,
            "parameters": [
                {
                    "component_type": "predictive_property",
                    "name": "classification",
                    "weight": 1,
                    "specific_parameters": {
                        "model_path": predictive_model_path,
                        "container_type": "scikit_container",
                        "uncertainty_type": None,
                        "scikit": "classification",
                        "transformation": {"transformation_type": "no_transformation"},
                        "descriptor_type": "ecfp_counts",
                        "size": 2048,
                        "radius": 2,
                        "use_counts": True,
                        "use_features": True,
                    },
                }
            ],
        },
    }

    return configuration


if __name__ == "__main__":
    main()


