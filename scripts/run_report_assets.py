from __future__ import annotations

from evaluate_conditions import main as evaluate_conditions_main
from check_project_ready import main as check_project_ready_main


def main() -> None:
    evaluate_conditions_main()
    check_project_ready_main()


if __name__ == "__main__":
    main()
