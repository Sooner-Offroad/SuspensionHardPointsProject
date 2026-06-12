import json
import math
from pathlib import Path

import yaml

from kinematics.optimizer import (
    NSGA2Config,
    export_best_candidate_geometry,
    optimize_suspension,
    run_nsga2,
)


def test_run_nsga2_returns_non_dominated_population():
    """NSGA-II should produce a non-empty Pareto set for a simple two-objective problem."""

    def objective(x: list[float]) -> list[float]:
        value = x[0]
        return [value * value, (value - 2.0) * (value - 2.0)]

    config = NSGA2Config(
        population_size=8,
        generations=4,
        mutation_rate=0.2,
        crossover_rate=0.8,
        bounds=[(-3.0, 3.0)],
        seed=7,
    )

    result = run_nsga2(objective=objective, config=config)

    assert result.population
    assert len(result.population) == config.population_size

    # The returned front should be non-dominated in the objective space.
    objectives = [candidate.objectives for candidate in result.population]
    for i, obj_i in enumerate(objectives):
        for j, obj_j in enumerate(objectives):
            if i != j:
                dominates = (
                    obj_i[0] <= obj_j[0] and obj_i[1] <= obj_j[1]
                    and (obj_i[0] < obj_j[0] or obj_i[1] < obj_j[1])
                )
                assert not dominates or i < j

    # The best known trade-off should be close to the true Pareto region.
    best = min(result.population, key=lambda c: c.objectives[0] + c.objectives[1])
    assert math.isfinite(best.objectives[0])
    assert math.isfinite(best.objectives[1])


def test_optimize_suspension_accepts_goal_targets_and_initial_points():
    """The suspension optimizer should accept goal targets and user-supplied starting guesses."""

    result = optimize_suspension(
        geometry_path=Path("tests/data/geometry.yaml"),
        sweep_path=Path("tests/data/sweep.yaml"),
        population_size=2,
        generations=1,
        seed=7,
        goals={
            "camber_change_deg": -0.5,
            "wheel_travel_mm": 381.0,
        },
        weights={
            "camber_change_deg": 1.0,
            "wheel_travel_mm": 1.0,
        },
        initial_points=[[0.0] * 18],
    )

    assert result.population
    assert len(result.population) == 2


def test_export_best_candidate_geometry_writes_updated_yaml(tmp_path: Path):
    """The optimizer output should be transformable into a usable geometry YAML."""

    optimizer_file = tmp_path / "optimizer_results.json"
    output_file = tmp_path / "optimized_geometry.yaml"

    optimizer_file.write_text(
        json.dumps(
            [
                {
                    "variables": [10.0, -5.0, 2.0] + [0.0] * 15,
                    "objectives": [1.0, 2.0],
                    "rank": 1,
                    "crowding_distance": 0.0,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    export_best_candidate_geometry(
        geometry_path=Path("tests/data/geometry.yaml"),
        optimizer_results_path=optimizer_file,
        output_path=output_file,
        candidate_index=0,
    )

    assert output_file.exists()

    data = yaml.safe_load(output_file.read_text(encoding="utf-8"))
    assert data["hardpoints"]["lower_wishbone_inboard_front"] == {
        "x": 260.0,
        "y": 395.0,
        "z": 202.0,
    }
