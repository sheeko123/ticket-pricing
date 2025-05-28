# Ticket Price Prediction Dashboard

A machine learning-powered dashboard for predicting and analyzing ticket prices for entertainment events. This project uses XGBoost models to predict ticket prices, with separate models optimized for floor and non-floor tickets.

## Features

- **Dual Model Approach**: Separate models for floor and non-floor tickets
- **Advanced Feature Engineering**: Temporal features, price normalization, demand-supply metrics
- **Interactive Dashboard**: Built with Streamlit for easy exploration
- **Price Trend Analysis**: Visualize price patterns and optimal purchase timing
- **Savings Calculator**: Calculate potential savings and ROI

## Model Performance

### Floor Tickets Model
- RMSE: $89.32
- R-squared: 0.8622
- Uses time series cross-validation

### Non-Floor Tickets Model
- RMSE: $16.12
- R-squared: 0.9314
- Uses stratified grouped cross-validation

## Project Structure

```
ticket_pricing/
├── src/
│   ├── models/
│   │   ├── dashboard_xgboost_models.py  # Model implementations
│   │   └── xgboost.py                   # Base model class
│   ├── etl/
│   │   └── features.py                  # Feature engineering
│   └── visualization/
│       └── plots.py                     # Plotting utilities
├── data/
│   ├── raw/                            # Raw data
│   └── processed/                      # Processed data
├── notebooks/
│   └── ticket_price_analysis.ipynb     # Analysis notebook
├── app.py                              # Streamlit dashboard
└── requirements.txt                    # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ticket-pricing.git
cd ticket-pricing
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit dashboard:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Key Features

### Model Performance
- View model accuracy metrics
- Compare actual vs. predicted prices
- Analyze feature importance

### Price Predictions
- Get price predictions for specific events
- View price trends over time
- Analyze price volatility

### Entry Price Dynamics
- Track price evolution
- Identify optimal purchase timing
- Calculate potential savings

### Savings Calculator
- Calculate potential savings
- Assess purchase timing
- Evaluate risk factors

## Development

### Adding New Features
1. Create a new branch
2. Implement your changes
3. Add tests if applicable
4. Submit a pull request

### Running Tests
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- XGBoost team for the powerful gradient boosting library
- Streamlit team for the excellent dashboard framework
- All contributors who have helped improve this project

## Contact

Dara Sheehan - [Your Email]

Project Link: [https://github.com/yourusername/ticket-pricing](https://github.com/yourusername/ticket-pricing) 