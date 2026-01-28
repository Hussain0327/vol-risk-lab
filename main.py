"""
Volatility Risk Lab - Main Pipeline

Compares volatility forecasting methods:
1. Rolling Window (naive baseline)
2. EWMA (exponential weighting)
3. GARCH (econometric)
4. ML Models (Ridge, Random Forest, GBM)

Evaluates using MAE, RMSE, and VaR backtesting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import ROLLING_WINDOWS, TRADING_DAYS_PER_YEAR
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models.ewma import EWMAVolatility
from src.models.garch import GARCHModel
from src.models.ml_vol import MLVolatilityModel
from src.risk_metrics import RiskMetrics
from src.evaluation import ModelEvaluator


def main():
    # ===================
    # 1. Load Data
    # ===================
    print("Loading data...")
    loader = DataLoader()
    df = loader.load()
    loader.validate()
    returns = loader.compute_returns().dropna()

    print(
        f"Data: {len(returns)} days from {returns.index.min().date()} to {returns.index.max().date()}"
    )

    # ===================
    # 2. Build Features
    # ===================
    print("\nBuilding features...")
    fe = FeatureEngineer(returns)

    # Target: 21-day forward realized volatility
    target = fe.realized_volatility(21)

    # Features for ML
    feature_matrix = fe.build_feature_matrix()
    macro_features = loader.get_features()
    features = pd.concat([feature_matrix, macro_features], axis=1)

    # ===================
    # 3. Train/Test Split
    # ===================
    # Use last 20% for testing
    split_idx = int(len(returns) * 0.8)

    returns_train = returns.iloc[:split_idx]
    returns_test = returns.iloc[split_idx:]

    target_test = target.iloc[split_idx:]

    print(f"Train: {len(returns_train)} days")
    print(f"Test: {len(returns_test)} days")

    # ===================
    # 4. Fit Models
    # ===================
    print("\n=== Fitting Models ===")

    # Model 1: Rolling Volatility (baseline)
    print("Fitting Rolling Volatility...")
    rolling_vol = fe.rolling_volatility(21)
    rolling_forecast = rolling_vol.iloc[split_idx:]

    # Model 2: EWMA
    print("Fitting EWMA...")
    ewma = EWMAVolatility(lambda_param=0.94)
    ewma.fit(returns_train)
    ewma_vol = ewma.get_volatility()
    # For test period, refit on expanding window
    ewma_full = EWMAVolatility(lambda_param=0.94)
    ewma_full.fit(returns)
    ewma_forecast = ewma_full.get_volatility().iloc[split_idx:]

    # Model 3: GARCH
    print("Fitting GARCH...")
    garch = GARCHModel(p=1, q=1)
    garch.fit(returns_train)
    print(f"GARCH params: {garch.get_params()}")
    garch_full = GARCHModel(p=1, q=1)
    garch_full.fit(returns)
    garch_forecast = garch_full.get_conditional_volatility().iloc[split_idx:]

    # Model 4: ML Models
    print("Fitting ML models...")

    # Prepare data
    ml_model = MLVolatilityModel(model_type="ridge")
    X_train, X_test, y_train, y_test = ml_model.prepare_data(features, target)

    # Ridge
    ridge = MLVolatilityModel(model_type="ridge")
    ridge.fit(X_train, y_train)
    ridge_forecast = pd.Series(ridge.predict(X_test), index=X_test.index)

    # Random Forest
    rf = MLVolatilityModel(model_type="rf")
    rf.fit(X_train, y_train)
    rf_forecast = pd.Series(rf.predict(X_test), index=X_test.index)

    # Gradient Boosting
    gbm = MLVolatilityModel(model_type="gbm")
    gbm.fit(X_train, y_train)
    gbm_forecast = pd.Series(gbm.predict(X_test), index=X_test.index)

    # ===================
    # 5. Evaluate Models
    # ===================
    print("\n=== Model Evaluation ===")

    evaluator = ModelEvaluator(target_test)
    evaluator.add_forecast("Rolling_21", rolling_forecast)
    evaluator.add_forecast("EWMA", ewma_forecast)
    evaluator.add_forecast("GARCH", garch_forecast)
    evaluator.add_forecast("Ridge", ridge_forecast)
    evaluator.add_forecast("RandomForest", rf_forecast)
    evaluator.add_forecast("GBM", gbm_forecast)

    results = evaluator.compare_all()
    results.to_csv("data/processed/model_comparison.csv")
    print("\nVolatility Forecast Accuracy:")
    print(results.round(4))

    # ===================
    # 6. VaR Backtesting
    # ===================
    print("\n=== VaR Backtesting ===")

    rm = RiskMetrics(returns_test)

    # Compute VaR using EWMA volatility
    ewma_var = -1.645 * ewma_forecast / np.sqrt(TRADING_DAYS_PER_YEAR)  # Daily VaR
    var_results = evaluator.var_backtest(returns_test, ewma_var, confidence=0.95)

    print(f"EWMA VaR Backtest:")
    print(f"  Breaches: {var_results['breaches']} / {var_results['total_days']}")
    print(f"  Breach rate: {var_results['breach_rate']:.2%}")
    print(f"  Expected rate: {var_results['expected_rate']:.2%}")
    print(f"  Ratio: {var_results['ratio']:.2f} (should be ~1.0)")

    # ===================
    # 7. Visualization
    # ===================
    print("\nGenerating plots...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Volatility forecasts vs realized
    ax1 = axes[0]
    target_test.plot(ax=ax1, label="Realized", color="black", linewidth=1.5)
    rolling_forecast.plot(ax=ax1, label="Rolling", alpha=0.7)
    ewma_forecast.plot(ax=ax1, label="EWMA", alpha=0.7)
    garch_forecast.plot(ax=ax1, label="GARCH", alpha=0.7)
    ax1.set_title("Volatility Forecasts vs Realized")
    ax1.set_ylabel("Annualized Volatility")
    ax1.legend()

    # Plot 2: ML model forecasts
    ax2 = axes[1]
    target_test.plot(ax=ax2, label="Realized", color="black", linewidth=1.5)
    ridge_forecast.plot(ax=ax2, label="Ridge", alpha=0.7)
    rf_forecast.plot(ax=ax2, label="RandomForest", alpha=0.7)
    gbm_forecast.plot(ax=ax2, label="GBM", alpha=0.7)
    ax2.set_title("ML Model Forecasts vs Realized")
    ax2.set_ylabel("Annualized Volatility")
    ax2.legend()

    # Plot 3: VaR breaches
    ax3 = axes[2]
    returns_test.plot(ax=ax3, label="Returns", color="blue", alpha=0.5)
    ewma_var.plot(ax=ax3, label="95% VaR", color="red", linewidth=1.5)

    # Mark breaches
    breaches = returns_test[returns_test < ewma_var]
    ax3.scatter(
        breaches.index, breaches.values, color="red", s=20, label="Breaches", zorder=5
    )
    ax3.set_title("VaR Breaches")
    ax3.set_ylabel("Return")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("data/processed/results.png", dpi=150)
    plt.show()

    # ===================
    # 8. Summary
    # ===================
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    best_model = results["MAE"].idxmin()
    print(f"\nBest model by MAE: {best_model}")
    print(f"MAE: {results.loc[best_model, 'MAE']:.4f}")

    print("\nKey findings:")
    print(
        "- "
        + (
            "EWMA/GARCH beat rolling baseline"
            if results.loc["EWMA", "MAE"] < results.loc["Rolling_21", "MAE"]
            else "Rolling baseline is competitive"
        )
    )
    print(
        "- "
        + (
            "ML models improve on traditional methods"
            if results.loc[["Ridge", "RandomForest", "GBM"], "MAE"].min()
            < results.loc[["Rolling_21", "EWMA", "GARCH"], "MAE"].min()
            else "Traditional methods hold their own"
        )
    )

    return results


if __name__ == "__main__":
    results = main()
