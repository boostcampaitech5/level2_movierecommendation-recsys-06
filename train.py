from utils.util import get_settings, get_raw_data


def main() -> None:
    settings: dict = get_settings()

    raw_user, raw_item, raw_interaction = get_raw_data(settings)

    return


if __name__ == "__main__":
    main()
