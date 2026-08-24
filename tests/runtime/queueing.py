from experiments.queueing import finite_window_littles_law


def main() -> None:
    records = [
        {"arrival_seconds": 0.0, "finished_offset_seconds": 2.0, "system_time_seconds": 2.0},
        {"arrival_seconds": 1.0, "finished_offset_seconds": 3.0, "system_time_seconds": 2.0},
    ]
    report = finite_window_littles_law(records, 3.0)
    assert report["method"] == "finite_window_arrival_departure_accounting"
    assert report["request_count"] == 2
    assert abs(report["residual"]) < 1.0e-12
    assert finite_window_littles_law([], 0.0)["request_count"] == 0
    print("queueing=pass")


if __name__ == "__main__":
    main()
