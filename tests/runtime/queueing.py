from experiments.queueing import (
    finite_window_littles_law,
    modeled_blocked_cohort_accounting,
)


def main() -> None:
    records = [
        {
            "arrival_seconds": 0.0,
            "finished_offset_seconds": 2.0,
            "system_time_seconds": 2.0,
        },
        {
            "arrival_seconds": 1.0,
            "finished_offset_seconds": 3.0,
            "system_time_seconds": 2.0,
        },
    ]
    report = finite_window_littles_law(records, 3.0)
    assert report["method"] == "finite_window_arrival_departure_accounting"
    assert report["request_count"] == 2
    assert abs(report["residual"]) < 1.0e-12
    assert finite_window_littles_law([], 0.0)["request_count"] == 0

    modeled = modeled_blocked_cohort_accounting(4, 100.0)
    assert modeled["pending_release_count"] == 4
    assert modeled["pending_area_unit_us"] == 200.0
    assert modeled["release_rate_per_second"] == 40_000.0
    assert modeled["mean_pending_units"] == 2.0
    assert modeled["mean_pending_us"] == 50.0
    assert "residual" not in modeled and "lhs" not in modeled and "rhs" not in modeled
    instant = modeled_blocked_cohort_accounting(4, 0.0)
    assert instant["pending_release_count"] == 4
    assert instant["release_rate_per_second"] == 0.0
    assert instant["mean_pending_units"] == 0.0
    print("queueing=pass")


if __name__ == "__main__":
    main()
