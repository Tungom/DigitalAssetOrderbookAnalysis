# Analyzing Orderbook Data for Digital Assets

This Repository contains the following files:

- requirements.txt: contains the libraries needed for this work
- DataUtils.py: classes and functions that help with preprocessing orderbook data
- OrderbookAnalysis.ipynb: The Notebook where orderbook analysis is done

The main goal here is to analyze cryptocurrency orderbook data, the impact of price and volume to identify key insights into market dynamics. We do this using Python, and our analytics process is laid out as follows:

1. **Data Acquisition and Preprocessing:** We use Bybit for this.
2. **Visualization:** We create visual charts to visualize price and volume changes.
3. **EDA:** We perform Exploratory Data Analysis on Price and Volume Dynamics.
4. **Market Patterns:** We do further analysis and feature extraction to find price patterns from volume and price changes.
5. **Price Prediction:** As a Bonus, we use XGBoost to learn a model that can predict future prices.

## Running the Notebook

To run the notebook locally:

1. Create a Conda environment with Python version 3.9 or above (recommended `conda create -n myenv python=3.9`)
2. Clone this repository.
3. Activate the Conda environment.
4. Navigate to the folder containing the `requirements.txt` file in your terminal.
5. Install the requirements by running `!pip install -r requirements.txt`.
6. Now you are ready to run the Notebook! Have Fun!!!

__Note:__ When downloading the data using the Bybit API, your IP address matters. The best locations are in the APAC region (use a VPN).
