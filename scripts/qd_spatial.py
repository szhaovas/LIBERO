import csv
import pickle as pkl
from functools import partial
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

task_5_bddl = (
    Path(get_libero_path("bddl_files"))
    / "custom"
    / "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl"
)
TASK_ENV = partial(
    OffScreenRenderEnv,
    bddl_file_name=task_5_bddl,
    camera_heights=256,
    camera_widths=256,
)


def evaluate(params, seed):
    """Evaluates params by creating LIBERO environments and computing
    objective and measure values from the environments' features and VLA
    rollout.

    Args:
        params (array-like): Array of shape (batch_size, solution_dim)
            containing params to be evaluated.

    Return:
        objective (np.ndarray): Array of shape (batch_size,) containing VLA's
            success rates on environments created from each row of ``params``.
            Invalid rows get np.nan.
        measures (np.ndarray): Array of shape (batch_size, measure_dim)
            containing measure values computed from the environments' features.
            Invalid rows get [np.nan] * measure_dim.
    """
    # NOTE: Ideally this should be parallelized, but we may have to evaluate
    # solutions one at a time due to VLA's high GPU memory requirement.

    objectives = []
    measures = []
    for sol in params:
        env = TASK_ENV(params=sol)

        env.seed(seed)
        try:
            env.reset()
        except ValueError as e:
            print(e)
            # TODO: How to handle solutions that fail to generate
            objectives.append(100)
            measures.append([0, 0])
            continue

        # TODO: right now compute_spread_similarity must be called at the start
        # before any action since actions might change objects' locations
        spread, similarity = env.env.compute_spread_similarity()
        measures.append([spread, similarity])

        # TODO: Get success rates by running openpi on env
        objectives.append(100)

    return np.asarray(objectives), np.asarray(measures)


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=100, cmap="viridis")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def main(
    iterations=1000,
    batch_size=16,
    num_emitters=1,
    seed=42,
    outdir="test_logs",
    log_every=10,
):
    logdir = Path(outdir)
    logdir.mkdir(exist_ok=True)

    # For now ``params`` should be an array listing object
    # coordinates in the following order:
    #   [
    #       akita_black_bowl_1_x, akita_black_bowl_1_y,
    #       akita_black_bowl_2_x, akita_black_bowl_2_y,
    #       cookies_1_x, cookies_1_y,
    #       glazed_rim_porcelain_ramekin_1_x,
    #       glazed_rim_porcelain_ramekin_1_y,
    #       plate_1_x, plate_1_y
    #   ]
    main_archive = GridArchive(
        solution_dim=10,
        dims=[100, 100],
        ranges=[(0, 1)] * 2,
        # learning_rate=0.1,
        # threshold_min=0,
        seed=seed,
    )
    passive_archive = GridArchive(
        solution_dim=10,
        dims=[100, 100],
        ranges=[(0, 1)] * 2,
        seed=seed,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=main_archive,
            # Range centers copied from BDDL file
            x0=[-0.18, 0.32, 0.13, -0.07, 0.07, 0.03, -0.20, 0.20, 0.06, 0.20],
            sigma0=0.02,
            # TODO: Define bounds if we want to stay close to the original BDDL
            bounds=None,
            batch_size=batch_size,
            seed=seed + i,
        )
        for i in range(num_emitters)
    ]

    scheduler = Scheduler(main_archive, emitters, result_archive=passive_archive)

    summary_filename = logdir / "summary.csv"
    with open(summary_filename, "w") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["Iteration", "QD-Score", "Coverage", "Maximum", "Average"])

    for i in range(1, iterations + 1):
        solutions = scheduler.ask()
        objectives, measures = evaluate(params=solutions, seed=seed)
        scheduler.tell(objectives, measures)

        print(
            f"\n------------------ Iteration{i} ------------------\n"
            f"\t QD-Score: {scheduler.archive.stats.qd_score}\n"
            f"\t Coverage: {scheduler.archive.stats.coverage}\n"
            f"\t Maximum : {scheduler.archive.stats.obj_max}\n"
            f"\t Average : {scheduler.archive.stats.obj_mean}\n"
        )

        final_itr = i == iterations
        if i % log_every == 0 or final_itr:
            pkl.dump(
                scheduler,
                open(logdir / f"scheduler_{i:08d}.pkl", "wb"),
            )

            with open(summary_filename, "a") as summary_file:
                writer = csv.writer(summary_file)
                data = [
                    i,
                    scheduler.archive.stats.qd_score,
                    scheduler.archive.stats.coverage,
                    scheduler.archive.stats.obj_max,
                    scheduler.archive.stats.obj_mean,
                ]
                writer.writerow(data)

            save_heatmap(
                scheduler.result_archive,
                logdir / f"heatmap_{i:08d}.png",
            )


if __name__ == "__main__":
    fire.Fire(main)
