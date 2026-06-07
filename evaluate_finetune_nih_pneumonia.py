import glob
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from cxrclip import seed_everything
from cxrclip.evaluator import Evaluator

log = logging.getLogger(__name__)


def print_evals(evals, metric="AUROC(Avg)", best="max"):
    keys = list(evals.values())[0].keys()
    st = "| model | " + " | ".join(keys) + "|\n"
    st += "| :---- | " + " | ".join([("-" * (len(k) - 1)) + ":" for k in keys]) + "|\n"

    best_score = 0.0 if best == "max" else 9e9

    for c, e in evals.items():
        filename = ".".join(c.split("/")[-1].split(".")[:-1])
        st += f"| {filename} | " + " | ".join([f"{e[k]:.3f}" for k in keys]) + " |\n"

        cur_score = e[metric]
        best_score = max(best_score, cur_score) if best == "max" else min(best_score, cur_score)

    st += f"Best {metric}: {best_score:.3f}\n"
    return st


@hydra.main(version_base=None, config_path="configs", config_name="eval_finetune")
def main(cfg: DictConfig):
    print(cfg.test.checkpoint)

    seed_everything(cfg.test.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    OmegaConf.resolve(cfg)

    if type(cfg.test.checkpoint).__name__ == "ListConfig":
        ckpt_paths = cfg.test.checkpoint
    elif os.path.isdir(cfg.test.checkpoint):
        ckpt_paths = sorted(glob.glob(os.path.join(cfg.test.checkpoint, "*.tar")))
    else:
        ckpt_paths = sorted(glob.glob(cfg.test.checkpoint))

    cfg_dict = OmegaConf.to_container(cfg)

    cfg_dict["data_test"] = {
        "nih_pneumonia": {
            "name": "nih_pneumonia",
            "data_type": "image_classification",
            "data_path": "/content/datasets/NIH_Extracted/csv/nih_pneumonia_test.csv",
            "n_class": 1,
            "normalize": "imagenet",
        }
    }

    cfg_dict["dataloader"]["test"]["batch_size"] = 64
    cfg_dict["dataloader"]["test"]["num_workers"] = 2
    cfg_dict["dataloader"]["test"]["prefetch_factor"] = 2

    evaluator = Evaluator(cfg_dict, ckpt_paths)
    save_path = os.path.dirname(ckpt_paths[0])

    for test_dataset_name in evaluator.test_dataloader_dict.keys():
        print(test_dataset_name)

        evals = {
            c: evaluator.evaluate_classifier(c, test_dataset_name)
            for c in ckpt_paths
        }

        multilabel_classification = {
            k: {
                _k: _v
                for _k, _v in v["multilabel_classification"].items()
                if not isinstance(_v, dict)
            }
            for k, v in evals.items()
        }

        st = print_evals(
            multilabel_classification,
            metric="AUROC(Avg)",
            best="max",
        )

        log.info(cfg.test.checkpoint)
        log.info(st)

        with open(os.path.join(save_path, f"results-{test_dataset_name}.txt"), "w") as outfile:
            outfile.write(st)


if __name__ == "__main__":
    main()
