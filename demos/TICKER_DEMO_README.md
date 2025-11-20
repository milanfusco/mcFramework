# Ticker Black-Scholes Analysis

Analyze any NYSE/NASDAQ stock ticker using Black-Scholes Monte Carlo simulations with market data!

## Quick Start

```bash
# Analyze Apple stock
python demoTickerBlackScholes.py AAPL

# Analyze Microsoft with custom parameters
python demoTickerBlackScholes.py MSFT --days 500 --horizon 2.0

# Analyze Tesla with more simulations
python demoTickerBlackScholes.py TSLA --simulations 5000 --seed 999
```

## What It Does

1. **Fetches Market Data** - Downloads historical stock prices from Yahoo Finance
2. **Estimates Parameters** - Calculates volatility and drift from market data
3. **Runs Simulations** - Performs Monte Carlo price forecasts using estimated parameters
4. **Prices Options** - Values call and put options using Black-Scholes
5. **Calculates Greeks** - Computes Delta, Gamma, Vega, Theta, and Rho
6. **Creates Visualizations** - Generates 6 comprehensive charts

## Generated Charts

1. **`historical_vs_simulated.png`** - Historical price data with Monte Carlo forecasts
   - Shows past performance
   - Overlays simulated future paths
   - Displays mean forecast and 90% confidence bands

2. **`return_distribution.png`** - Comparison of historical vs simulated returns
   - Validates model calibration
   - Side-by-side histograms

3. **`forecast_distribution.png`** - Distribution of forecasted prices
   - Shows probability of different outcomes
   - Displays percentiles and statistics

4. **`option_analysis.png`** - Complete option pricing analysis
   - Option price vs stock price
   - Greeks bar chart
   - Sensitivity to volatility
   - Time decay curve

5. **`call_surface_3d.png`** Call option surface plot
   - Shows call option prices across stock prices and time
   - Uses "viridis" colormap (green-blue gradient)
   - Black paths show simulated price trajectories
   - Red mean path highlights the average trajectory

6. **`put_surface_3d.png`** Put option surface plot
   - Shows put option prices across stock prices and time
   - Uses "plasma" colormap (purple-orange gradient)
   - Dark red paths show simulated trajectories
   - Blue mean path highlights the average trajectory

## Command Line Options

```
positional arguments:
  ticker                Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)

options:
  -h, --help            show this help message and exit
  --days DAYS           Number of historical trading days to fetch (default: 252)
  --horizon HORIZON     Forecast horizon in years (default: 1.0)
  --simulations SIMULATIONS
                        Number of Monte Carlo simulations (default: 1000)
  --paths PATHS         Number of paths to visualize (default: 100)
  --seed SEED           Random seed for reproducibility (default: 42)
```

## Examples

### Tech Stocks
```bash
python demoTickerBlackScholes.py AAPL   # Apple
python demoTickerBlackScholes.py MSFT   # Microsoft
python demoTickerBlackScholes.py GOOGL  # Google
python demoTickerBlackScholes.py NVDA   # NVIDIA
python demoTickerBlackScholes.py META   # Meta
```

### Financial Sector
```bash
python demoTickerBlackScholes.py JPM    # JPMorgan Chase
python demoTickerBlackScholes.py BAC    # Bank of America
python demoTickerBlackScholes.py GS     # Goldman Sachs
```

### Long-Term Analysis
```bash
# 5-year forecast with 2 years of historical data
python demoTickerBlackScholes.py AAPL --days 504 --horizon 5.0 --simulations 5000
```

## Output

All charts are saved to: `img/{TICKER}_BlackScholes/`

Example output for AAPL:
```
img/AAPL_BlackScholes/
├── historical_vs_simulated.png
├── return_distribution.png
├── forecast_distribution.png
└── option_analysis.png
```

## Requirements

- `yfinance` - For fetching stock data (automatically installed)
- Active internet connection

## How It Works

### Parameter Estimation
The demo estimates Black-Scholes parameters from historical data:

- **Current Price (S₀)**: Latest closing price
- **Volatility (σ)**: Annualized standard deviation of log returns
- **Drift (μ)**: Annualized mean return

### Option Pricing
Uses risk-neutral pricing with:
- Strike price (K) = Current price (ATM option)
- Time to maturity (T) = 3 months
- Risk-free rate (r) = 5% annual

### Forecasting
Simulates future stock prices using Geometric Brownian Motion:
```
dS_t = μ S_t dt + σ S_t dW_t
```

## Tips

1. **More historical data** = Better parameter estimates
   - Use `--days 504` for 2 years of data

2. **More simulations** = Smoother distributions
   - Use `--simulations 5000` for production analysis

3. **Reproducibility** - Use same `--seed` for consistent results

4. **Compare tickers** - Run multiple tickers and compare volatilities

## Limitations

- Assumes log-normal returns (may not hold for all stocks)
- Historical volatility may not predict future volatility
- Black-Scholes assumptions (constant volatility, no dividends, etc.)
- Past performance doesn't guarantee future results

---

**Disclaimer**: This is for educational purposes only. Not financial advice!
