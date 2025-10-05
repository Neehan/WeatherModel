import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.crop_yield.dataloader.numpy_yield_dataloader import get_numpy_train_test_data
from src.crop_yield.dataloader.yield_dataloader import (
    read_usa_dataset,
    read_non_us_dataset,
)
from src.utils.constants import (
    DATA_DIR,
    CROP_YIELD_STATS,
    EXTREME_YEARS,
    TEST_YEARS,
)


def random_forest_yield_training_loop(args_dict, use_cropnet: bool) -> dict:
    logger = logging.getLogger(__name__)

    if use_cropnet:
        raise ValueError("CropNet not supported for Random Forest baseline")

    country = args_dict["country"]
    crop_type = args_dict["crop_type"]
    n_train_years = args_dict["n_train_years"]
    n_past_years = args_dict["n_past_years"]
    test_type = args_dict.get("test_type", "extreme")

    # Read dataset
    if country == "usa":
        crop_df = read_usa_dataset(DATA_DIR)
    else:
        crop_df = read_non_us_dataset(DATA_DIR, country)

    # Get test years
    if test_type == "extreme":
        test_years = EXTREME_YEARS.get(country, {}).get(crop_type)
        if test_years is None:
            raise ValueError(f"No extreme years found for {crop_type} in {country}.")
    elif test_type == "overall":
        test_years = TEST_YEARS
    elif test_type == "ahead_pred":
        test_years = TEST_YEARS
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    # Override test year if provided
    if args_dict.get("test_year") is not None:
        test_years = [args_dict["test_year"]]

    logger.info(
        f"Random Forest training on {crop_type} in {country}, test years: {test_years}"
    )

    fold_rmses = []

    for test_year in test_years:
        logger.info(f"Training fold for test year {test_year}")

        test_gap = 4 if test_type == "ahead_pred" else 0

        # Get numpy data
        (X_train, y_train), (X_test, y_test) = get_numpy_train_test_data(
            crop_df,
            n_train_years,
            test_year,
            n_past_years,
            crop_type,
            country,
            test_gap=test_gap,
        )

        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train Random Forest with hyperparameters from config
        # Must match baseline_grid_search.py exactly
        model = RandomForestRegressor(
            n_estimators=args_dict["n_estimators"],
            max_depth=args_dict["max_depth"],
            min_samples_leaf=args_dict["min_samples_leaf"],
            max_features="sqrt",  # type: ignore
            random_state=args_dict["seed"],
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Compute RMSE
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        fold_rmses.append(rmse)

        logger.info(f"Test year {test_year}: RMSE = {rmse:.4f}")

    return {
        "fold_results": fold_rmses,
        "avg_rmse": np.mean(fold_rmses),
        "std_rmse": np.std(fold_rmses),
    }
