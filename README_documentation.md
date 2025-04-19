# Project Title: ON-CHAIN DATA AUTOMATED TRADING MACHINE
## Overview
This project focuses on applying statistical-based Hidden Markov Models(HMM), Natural Language Processing(NLP) that act as indicators/ filters that enhance the entire backtest process, in order to boost profitability by hitting a higher sharpe ratio while achieving the criteria for maximum drawdown and trade frequency.
In case the slides pdf is unusable: https://www.canva.com/design/DAGkx0Caa0c/Pt7mE2aNObOEtgHcccmw2Q/editutm_content=DAGkx0Caa0c&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Tools Used
- Python
- Jupyter Notebook
- Multiple machine learning Libraries
- Graphing Libraries
- Statistical tools
  
## Architecture Workflow

### HMM Backbone
  #### Part 1 - Obtaining Data and Identifying Basic Relationships
  - Features input for visualisation (from distinct endpoints, merged)
    
    ```
     📌 Data pre-stored in CSV to reduce redundant read operations
     📌 Exchange flow endpoints prioritized: flow_mean, flow_total, transaction_count
     📌 Additional metrics: inflow, outflow, netflow 
     📌 Handling missing values and synthetic data generation:
         👉 f_ttl = concat(r1, r2) + exponential noise
         👉 f_mean = f_ttl / uniform(10–30)
         👉 t_cnt = f_ttl * rand(0.5–2) + base offset
     📌 Aggregated into hourly intervals, covering 5 years
     📌 5-year visualisation using timeseries plots
    ```
    
  - Correlation tables between features
    
    ```
     📌 Generate correlation matrix (0<=x<=1) among features
     📌 Analyse the relationship between features
     📌 Visualised using heatmaps for feature prioritisation
    ```
    
  - Frequency plots against features
    
     ```
     📌 Identifying the norm of the crypto, a significant signal for upcoming actions
       - Verify whale dominations 
       - Verify whale density
       - Verify the presence of speculative bubbles
     📌 Enable data-driven decision-making through graph shapes
       - Ensure the active trading environment
    ```
  
  #### Part 2 - Optimising Model Selection
  - Model selection using Bayesian Information Criterion, Akaike Information Criterion 
     ```
        Statistical approach:
        n_features = X_scaled.shape[1]
        n_params = n_states * n_states + n_states * n_features + n_states * n_features * (n_features + 1) // 2
        log_likelihood = model.score(X_scaled)
        bic = -2 * log_likelihood + n_params * np.log(X_scaled.shape[0])
        aic = -2 * log_likelihood + 2 * n_params
    ```
  - Silhouette score to evaluate the quality of clustering
    ```
       silhouette = silhouette_score(X_scaled, hidden_states)
       silhouette_scores.append(silhouette)
    ```
  - Elbow Plots which visualize the minimization process, state prediction
    ```
      📌 Append BIC and AIC score to an array
      📌 Identify the model(distinct number of states)that have lowest relative score
      📌 Visualise Silhouette score to identify how well the data fits the cluster
    ```
  - Regime Classification and Distribution Plots
    ```
      📌 Statistically backed regime classification for data
      📌 5-year time series datapoint Visualisation
         - Enable us to easily identify data with extreme flow means
         - Better understanding of regime characteristics
      📌 Visualise Silhouette score to identify how well the data fits their cluster
    ```
  - Summary Metrics for Further Regime Characteristic Identifications
    ```
    Example(using flow mean metric):    
    regime         mean         std        min          max                                                            
      0         7.288524    3.084889   0.006716    15.920316   
      1        20.494636    6.007822   0.089733    38.411333      
      2       216.062946  352.265007   0.023389  4447.247543   
      3         2.868473    1.706804   0.000000     7.368699      
      4        53.256369   20.854317  24.866651   144.325671     
      5        13.094486    8.696826   0.064580    95.027511

    📌 Did for all metrics as above
    📌 Bar Chart Graphs Visualisation to identify outlying regimes for each metric
    📌 Obtain Insights for future in-depth regime analysis
    ```
  #### Part 3 - Regime Transition Handling
  - Keeptrack regime transition by count
     ```
      Regime transition counts:
          To   0     1    2     3     4     5
      From                                   
      0       7312  2710  349  4235  1884  2012
      1       3003  2206  137  1609  1153  1130
      2        276   174  153   152   242   177
      3       3813  1481  214  2980  1245  1075
      4       1910  1291  183  1174  1655   767
      5       2188  1376  138   659   801  1320
        
      📌 Verify how likely systems move from one regime to another
      📌 Powerful tool during the backtest period
      📌 Empowers prediction by forecasting regimes, identifying dominant transition paths
      ```
  - Convert to regime transition probability
     ```
      📌 Similar to counts, easier to use for probabilistic modelling or as transition matrices in HMMs
     ```
  - Regime Interpretation based on metrics
    ```
      📌 Each regime is characterized by statistical properties: Stability, Duration, Frequency
      📌 Enables semantic understanding, the representation of the regime's plain text meaning
      📌 Essential for drawing meaningful insights from clusters
    ```
  #### Part 4 - Transition Precision Simulation
   - Train and Test Set Splitting from Original Datasets （supervised learning)
     ```
      📌 The original dataset was split into two: the train set and the test set
      📌 Learn how well a model can identify the current regime 
     ```
   - Visualisations to indicate precision
       ```
      📌 Time series plots overlaid with predicted vs. true regimes
      📌 Learn how well a model can identify the current regime 
     ```
   - Regime Prediction Accuracy
       ```
      📌 Match predicted regime sequences with actual using mapping logic
      📌 Compute aligned accuracy score to evaluate label consistency
      📌 Helps quantify model performance across all regimes (not just the majority class)
     ```
   - Regime Transition Detection Accuracy
       ```
      📌 Focus on identifying regime *switch points* (transitions), not just static regimes
      📌 Evaluate using precision, recall, and F1 for transition detection 
     ```

 
### NLP Support
  #### Part 1 - Preprocessing and Feature Engineering
   - Input preprocessing and representation
     ```
     📌 Data loaded from structured CSV or DataFrame  
     📌 Pre-labelled with sentiments: negative, neutral, positive  
     📌 Normalization includes lowercasing, punctuation removal  
     📌 Token and word-level handling for varied sources (tweets, news, posts)
     ```
   - TF-IDF Feature Construction
     ```
     📌 Use TF-IDF Vectorizer with unigram and bigram support  
     📌 Converts text to sparse numerical vectors  
     📌 Ideal for small-to-medium datasets  
     📌 Sample weighting is used to address class imbalance  
     ```
   - Label Encoding
     ```
     📌 Sentiment class labels mapped for modelling:
        👉 ‘negative’ → 0
        👉 ‘neutral’  → 1
        👉 ‘positive’ → 2
     📌 Consistent across both traditional ML and BERT pipelines  
     ```
  #### Part 2 - Model Building and Optimisation
   - Logistic Regression with TF-IDF
     ```
     📌 Classic ML pipeline: TF-IDF + Logistic Regression  
     📌 Class weights handled to reduce bias from imbalanced data
     📌 Fast training, interpretable coefficients  
     📌 Good baseline for resource-constrained environments  
     ```
   - BERT (DISTILBERT) Fine-Tuning
     ```
     📌 Tokenization via HuggingFace `DistilBertTokenizer`  
        👉 Pad and truncate to 128 tokens  
        👉 Attention masks generated  
     📌 Sentiment classification head with 3 output neurons  
     📌 Fine-tuning parameters:  
        👉 Mixed precision (FP16)  
        👉 Gradient accumulation  
        👉 Epoch-based model checkpointing  
     ```
   - Metrics for Evaluation
     ```
     📌 Evaluation using:
        👉 Accuracy Score  
        👉 Weighted F1 Score  
     📌 Track performance on the test set per epoch  
     📌 Highlighted per model: traditional vs. transformer-based  
     ```
  #### Part 3 - Comparative Visualisation and Analysis
   - Score Tracking and Comparison
     ```
     📌 Evaluation matrix for both models:
         - Accuracy comparison (TF-IDF vs. BERT)  
         - F1 score: highlighting class prediction quality 
     ```
  #### Part 4 - Future Directions and Production-Level Strategy
   - Domain-Specific Model Integration
     ```
     📌 Integration of `ProsusAI/finbert` for financial language  
     📌 Domain-tuned BERT boosts performance in crypto finance context  
     📌 Transfer learning from FinBERT to enhance downstream tasks  
     ```
   - Hybrid Ensemble Model (Future Work)
     ```
     📌 Combine predictions from:
        - TF-IDF model (fast, stable)
        - BERT model (deep, contextual)
     📌 Use ensemble logic or meta-classifier for final decision     
     ```
   - Deployment Considerations
     ```
     📌 TF-IDF pipeline suitable for lightweight deployment  
     📌 BERT pipeline suited for heavyweight deployment (the better model)
     📌 REST API with Flask/FastAPI backend for real-time sentiment scoring  
     ```
     

### Data Manipulation
#### Part 1 - Initial Data Standardization and Cleaning
   - Loads Raw Data (‘output_data_with_regime.csv’)
   - Apply Datetime Rounding - rounding seconds (output_data_with_regime)
      ```
        if seconds >= 30, increases minute by 1;
        Set seconds and microseconds to 0
        Return formatted string
      ```
   - Handling Errors
      ```
       📌 return original string and print a warning for invalid & missing values cases
       📌 use try-except flow statement in handling exceptions
      ```
   - Save Cleaned Data Outputs (‘cleaned_cryptoquant_data.csv’)
    
#### Part 2 - Datetime Alignment and Merging
   - Parse Datetime Strings
      ```
       📌 Convert datetime string columns to datetime objects
      ```
   - Apply Datetime Rounding (BTCUSD)
      ```
       📌 Align Timezones to UTC
       📌 Ensures both datasets’ key datatime columns are timezone-aware and set to UTC, preventing mismatches during merging
      ```
   - Merge Dataframes
      ```
       📌 Keep only matching datetime
       📌 Add suffixes to distinguish overlapping column names
      ```
   - Save Merged Data (merged_crypto_btcusd_data.csv)
     
#### Part 3: Final Cleaning, Validation, and Structuring
   - Remove Redundant Columns (‘df.drop’)
   - Delete intermediate columns created during parsing and merging
   - Standardize Column Names for clarity(‘df.rename’)
   - Consolidate Timestamps
     ```
     📌 Compares auxiliary timestamp columns (‘start_time, ‘Timestamp’) if present.
     📌  Potentially drops ‘Timestamp’ if deemed redundant based on average difference from ‘start_time
     ```
   - Ensure Data Types
     ```
     📌  Confirms final ‘datetime’ column is a datetime object
     📌  Rounds numerical column (OHLCV, flow metrics) to standard decimal places
     ``` 
   - Validate Data Integrity
      ```
      📌 Checks for & handles potential issues by:
         👉Remove Duplicates (‘df.duplicated’, df.drop_duplicates’)
         👉Check for Missing Values (‘df.isnull().sum’) and prepare for interpolation.
         👉Check Internal Consistency
      ```
   - Structure Data (‘df.sort_values’, column reordering)
      ```
       📌 Sorts dataset chronologically by the main ‘datetime’ column
      ```
   - Save Final Data (‘cleaned_bitcoin_data.csv)

### Features Engineering

   - Inflow / Outflow ratios
     ```
       👉inflow_outflow_ratio
        Calculation: “Total Inflow / Total Outflow”
        Purpose: Indicates the balance of coins moving onto exchanges (potential selling pressure) versus off exchanges (potential buying/HODLing pressure). 
        Extreme values or sharp changes could signal shifts in market sentiment or supply dynamics.
        Ratio > 1 suggests more inflow, while < 1 suggests more outflow.
        Potential signal: 
        High values might precede price drops; low values might precede price increases

        👉inflow_outflow_ratio_log
        Calculation: ln (1 + inflow_outflow_ratio), clipped at 0
        Purpose: Normalizes the distribution of the `inflow_outflow_ratio`, making it potentially more suitable as an input for models sensitive to feature       
        scaling. Reduces the impact of extreme outliers while preserving the directional information.
        Potential signal: 
        Similar but with potentially better behavior
      ```

  - Top wallet concentration
    
      ```
      
        👉top10_dominance_inflow
        Calculation: Top 10 Inflow / Total Inflow
        Purpose: Measures the proportion of total exchange inflows originating from the 10 largest inflow transactions (proxy for whales). A high value indicates 
        that large players dominate inflows, potentially signaling large sell-offs or strategic positioning.
        Potential_signal: 
        Sudden increases might indicate whale selling preparation

        👉Top10_dominance_outflow
        Calculation: Top 10 Outflow / Total Outflow
        Purpose: Measures the proportion of total exchange outflows going to the 10 largest outflow transactions. A high value suggests whales are withdrawing 
        significant amounts, potentially for long-term holding or accumulation.
        Potential signal: 
        Sudden increases might indicate whale accumulation.
      
       ```


  - Net flow features
    
       ```
       
          👉netflow_sign
          Calculation: Sign (+1, -1, 0) of Total Netflow
          Purpose: Provides a clear, simple indicator of the immediate net direction of coin movement relative to exchanges (positive = net inflow, negative = net 
          outflow). Capture the prevailing short-term supply pressure.
          Potential signal: 
          Consistent positive signs may indicate selling pressure; consistent negative signs may indicate buying/holding pressure

         👉netflow_to_volume
         Calculation: Total Netflow / Trading Volume
         Purpose: Relates the net amount moving to/from exchanges to the overall trading activity. A high absolute value suggests that exchange flows are 
         significant relative to trading volume, potentially having a larger price impact.
         Potential signal: 
         large positive/negative values coinciding with price moves might confirm flow-driven pressure

         👉netflow_to_volume
         Calculation: Total Netflow / Trading Volume
         Purpose: Relates the net amount moving to/from exchanges to the overall trading activity. A high absolute value suggests that exchange flows are       
         significant relative to trading volume, potentially having a larger price impact.
         Potential signal: 
         large positive/negative values coinciding with price moves might confirm flow-driven pressure.
       
        ``` 


 - Transaction count features
   
        ``` 
           👉tx_count_per_volume
           Calculation: Transaction Count / Trading Volume
           Purpose: Compares the frequency of on-chain exchange-related transactions to the trading volume. A high value might indicate many small transactions 
           relative to volume (potentially retail activity), while a low value might suggest fewer, larger transactions or high volume driven by off-chain 
           activity.
           Potential signal: 
           Changes in this ratio could signal shifts in market participation structure (retail vs. institutional).
         ```

 - Flow rate changes
   
         ```
   
           👉flow_mean_change
            Calculation: Percentage change in the mean flow value from the previous period.
            Purpose: Measures the short-term change in the average size of transactions involved in exchange flows. An increasing value indicates larger average 
            flow sizes, potentially signaling increased conviction or larger players becoming active.
            Potential signal: 
            Sharp increases/decreases could precede volatility or trend changes.
   
         ``` 
         
- Lagged features
  
        ``` 
          👉inflow_outflow_ratio_lag1, netflow_total_lag3, top10_dominance_inflow_lag7
          Calculation: Value of the base feature shifted back by 'lag' periods (1, 3, or 7).
          Purpose: Provides the model with historical context for key metrics. Allows the model to learn patterns based not just on the current value but also on 
          recent past values, potentially capturing momentum or autoregressive effects.
          Potential signal: 
          Comparison between current and lagged values can indicate acceleration/deceleration or divergence.
        
        ``` 
 - Flow acceleration

         ```
   
           👉inflow_acceleration
            Calculation: Second difference of Total Inflow
            Purpose: Measures the rate of change of the *change* in total inflow. Positive acceleration means inflows are increasing at a faster rate (or 
            decreasing at a slower rate). Negative acceleration means the opposite. Captures shifts in the momentum of inflows.
            Potential signal: 
            Peaks/troughs in acceleration might precede turning points in inflow trends or price.
  
            👉outflow_acceleration
            Calculation: Second difference of Total Outflow
            Purpose: Measures the acceleration of total outflow. Captures shifts in the momentum of outflows, similar to inflow acceleration.
            Potential signal: 
            Similar to inflow acceleration, could signal turning points
           
        ```
- Moving average features of on-chain metrics
       ``` 
          👉e.g. , inflow_total_ma7, netflow_total_30
          Calculation: Rolling moving average of the base feature over a specified window (7, 14, or 30).
          Purpose: Smooths out short-term fluctuations in the base metrics, revealing underlying trends in inflows, outflows, or netflows over different time 
          horizons (short, medium, long).
          Potential signal: 
          Crossovers between different MA windows or between the metric and its MA can generate trend signals.
      ``` 
 - Volatility in on-chain metrics
     ``` 
         👉inflow_volatility 
         Calculation: Rolling 7-period Coefficient of Variation (Std Dev / Mean) of Total Inflow.
         Purpose: Measures the *relative* volatility of exchange inflows compared to their recent average size. High values indicate erratic or unstable inflow 
         patterns, while low values suggest consistency.
         Potential signal: 
         Periods of low volatility might precede breakouts; high volatility might signal uncertainty or distribution/accumulation climaxes.
  
         👉outflow_volatility
         Calculation: Rolling 7-period Coefficient of Variation (Std Dev / Mean) of Total Outflow.
         Purpose: Measures the relative volatility of exchange outflows. Similar interpretation to inflow volatility but for coins leaving exchanges.
         Potential signal: 
         Similar signals to inflow volatility, potentially indicating accumulation stability or distribution panic.
       ``` 
  - Divergence between inflow and outflow volatility
       ``` 
         👉flow_divergence
          Calculation: Absolute difference between `inflow_volatility` and `outflow_volatility`.
          Purpose: Highlights periods where the stability patterns of inflows and outflows differ significantly. For example, stable outflows but volatile inflows 
          might suggest panic selling or uncertain buying interest.
          Potential signal: 
          High divergence could signal market confusion or transitional phases.
        ``` 
  - On-chain trend features (based on 7-day change)
        ``` 
          👉inflow_trend
          Calculation: Binary indicator (1 if 7-period % change in Total Inflow > 0, else 0).
          Purpose: Provides a simple binary signal indicating the direction of the short-term (weekly) trend in exchange inflows.
          Potential signal: 
          persistent 1s suggest increasing selling pressure; persistent 0s suggest easing pressure.
  
          👉outflow_trend
          Calculation: Binary indicator (1 if 7-period % change in Total Outflow > 0, else 0).
          Purpose: Provides a binary signal for the direction of the short-term (weekly) trend in exchange outflows.
          Potential signal: 
          persistent 1s suggest increasing withdrawal/HODLing;
          persistent 0s suggest less withdrawal activity.

 - Trend alignment (Are inflow and outflow trends moving together?)
      ``` 
        👉trend_alignment
        Calculation: Binary indicator (1 if `inflow_trend` == `outflow_trend`, else 0).
        Purpose: Indicates whether the short-term trends of inflows and outflows are moving in the same direction (alignment = 1) or opposite directions 
        (alignment = 0). Divergence (0) might signal market indecision or turning points.
        Potential signal: 
       Shift from 1 to 0 or vice versa
      ``` 
- On-chain momentum (Short MA / Long MA ratio)
     ``` 
        👉inflow_momentum
        Calculation: Ratio of short-term MA (7) to long-term MA (30) for Total Inflow, centered around 0.
        Purpose: Measures the momentum of exchange inflows. Positive values indicate the short-term average inflow is higher than the long-term average (upward            momentum), negative values indicate the           opposite.
        Potential signal:
        Crossovers through 0 or extreme positive/negative values could signal shifts in inflow strength.
  
        👉outflow_momentum
        Calculation: Ratio of short-term MA (7) to long-term MA (30) for Total Outflow, centered around 0.
        Purpose: Measures the momentum of exchange outflows. Interpreted similarly to inflow momentum but for coins leaving exchanges.
        Potential signal: 
        Similar signals to inflow momentum, reflecting strength of withdrawals/accumulation.
     ``` 
 - Exchange reserve metrics
      ``` 
        👉cumulative_netflow
        Calculation: Cumulative sum of (inflow_total - outflow_total)
        Purpose: Tracks the overall historical net balance change on exchanges. Provides a longer-term perspective on supply availability on trading platforms.
        Potential signal: 
        Long-term trends (increasing/decreasing) can indicate macro accumulation or distribution phases

        👉reserve_change_rate
        Calculation: 7-day percentage change of cumulative_netflow
        Purpose: Measures the recent weekly rate of change in the overall exchange balance trend. Highlights acceleration or deceleration in net exchange supply 
        shifts
        Potential signal: 
        Sharp increases/decreases could signal imminent shifts in supply/demand balance.
       ``` 

  - Whale transaction metrics
      ``` 
        👉whale_netflow
        Calculation: inflow_top10 - outflow_top10
        Purpose: Isolates the net flow direction specifically for large ('whale') transactions. A positive value suggests whales are net senders to exchanges;     
        negative suggests they are net withdrawers.
        Potential signal: 
        often considered a leading indicator
        can precede price action
  
        👉whale_activity
        Calculation: Average of inflow_top10 and outflow_top10
        Purpose: Gauges the overall volume of large transactions interacting with exchanges, regardless of direction. High activity indicates significant whale            movement.
        Potential signal: 
        Increased activity might precede periods of higher volatility or trend changes driven by large players.
     ``` 
 - Whale Dominance Trend
     ``` 
        👉whale_dominance_trend
        Calculation: average of top10_dominance_inflow and top10_dominance_outflow
        Purpose: Provides a smoothed measure of the proportion of total exchange flows attributable to whales.
        Potential signal: 
        Trend can indicate whether market movements are increasingly driven by large players / becoming more distributed.

        👉whale_dominance_change
        Calculation: 3-day percentage change of whale_dominance_trend
        Purpose: Measures the short-term change in whale influence. Rapid changes might signal shifts in market control or imminent large player actions.
        Potential signal: Spikes could precede whale-driven volatility
     ```
 - Transaction activity metrics
     ```
        👉tx_momentum
        Calculation: 7-day percentage change of transactions_count_flow
        Purpose: Measures the weekly momentum of overall exchange-related transaction frequency
        Potential signal: 
        Increasing momentum can suggest growing network usage or speculative interest.
  
        👉tx_acceleration
        Calculation: 1-day difference (change) of tx_momentum
        Purpose: Measures the acceleration/deceleration of transaction frequency momentum
        Potential signal: 
        Positive acceleration suggests activity is picking up speed; negative suggests it's slowing down.
  
        👉tx_volatility
        Calculation: Rolling 7-day Coefficient of Variation of transactions_count_flow
        Purpose: Measures the relative stability of transaction counts. High volatility suggests inconsistent activity.
        Potential signal: 
        Low volatility might indicate a stable market phase; increasing volatility could precede price moves.
    ```
 - Market state classification
     ```
        👉accumulation_phase
        Calculation: Binary flag based on inflow/outflow ratio and its MA being below thresholds.
        Purpose: Heuristic attempt to identify potential accumulation periods based on strong net outflows from exchanges.
        Potential signal: 
        '1' suggests conditions often associated with market bottoms or consolidation before upward moves.
  
        👉distribution_phase
        Calculation: Binary flag based on inflow/outflow ratio and its MA being above thresholds
        Purpose: Heuristic attempt to identify potential distribution periods based on strong net inflows to exchanges.
        Potential signal: 
        '1' suggests conditions often associated with market tops or selling pressure building up.
      ```
###  Core
 - Constructor
   1. Data Validations
     ```
      If df is empty -> abort
      Make sure predictions is a pandas Series and aligns its index to match df.index
      Fills any missing predictions with NaN with neutral value 0.5
     ```
   2. Prediction Handling
    ```
      Adds predictions to the dataframe under self.df['prediction']
    ```
   3. Column Check
     ```
        Ensures the price_col (e.g. 'close') exists in the dataframe
     ```
   4. Running the Strategy
     ```
       This is where the core logic of buy/hold/sell happens -> self._run_strategy()
      This result is saved in self.results
     ```
  - Strategy
   1. Setup and Initialization
     ``` 
         Copy input DataFrame to avoid modifying original
         Initialize columns in DataFrame and check for required columns
     ```
   2. Main Backtest Loop
     ```
        get current_price and ML prediction for that day
        skip the step if either is NaN
        if prediction > 0.55: Strong Bullish -> Buy (1)
        if prediction < 0.45: Strong Bearish -> Sell (-1)
        Else (0.45 ≤ prediction ≤ 0.55): Hold (0)
        Carry over previous values
     ```
   3. Trade Execution Logic
      ```
        (i) Buy Logic
        if ML signal is 1
        Calculate how much BTC to buy using BASE_POSITION_SIZE
        Deduct trade amount and transaction fee from cash
        Add BTC bought to holdings
        Calculate new average entry price (prev_btc_held * prev_entry_price) + (btc_to_buy * current_price)/total_btc
        Set position = 1 and position_signal = 1
        Mark a buy trade (trade = 1), store entry_date

        (ii) Hold Logic
        signal is 0, just carry forward values
        No changes to cash, btc_held, or portfolio values

        (iii) Sell Logic
        Signal is -1 and holding BTC
        Sells 50% of current holdings (partial sell approach)
        Calculate fees and cash return
        If remaining BTC > 0: position = 1, position_signal = 0.5
        If no BTC remains: position = 0, position_signal = 0, reset entry_price
        Record trade result (win/loss)
        Calculate PnL percentage: ((current_price / prev_entry_price) - 1) * 100
        Store trade history with details about partial selling
      ```

  4. Key Formula Implementation
      ``` 
       Calculate trade_size = absolute change in position
       PnL calculation: (price_change * prev_position_signal) - (trade_size * transaction_fee)
       Where price_change = (current_price / previous_price) - 1
       This scales profit/loss by position signal strength (0-1)
       Calculate portfolio value = current_cash + (current_btc_held * current_price)
     ```

   5. Post Loop Calculations
       ```
        daily_return = portfolio_value.pct_change()
        cumulative_return = (portfolio_value / initial_cash) - 1
        Final trade_history with partial sell information
      ```
      
 - Calculate_metrics
    1. Return Calculations
       ```
        Total return = (final_value / initial_cash) - 1
        Annual return = total_return / (len(results) / (365 * 24)) for hourly data
       ```
    2. Sharpe Ratio Calculation (comparing portfolios or strategies over time)
       ```
          Formula: (pnl_mean / pnl_std) * √(365 * 24)
          Uses hourly adjustment factor √(365 * 24)
          Returns 0 if insufficient data or zero standard deviation
       ```
    3. Max Drawdown: How far the portfolio fell from its peak
       ```
         Formula: Drawdown(t) = (Portfolio_Value(t) - Rolling_Max(t)) / Rolling_Max(t)
         Max_Drawdown = minimum value of all drawdowns
         Uses safe_rolling_max with NaN handling to avoid division by zero
       ```
    4. Trades and Win Rate
       ```
        Total trades = (trade_size > 0).sum()
        Win rate = positive_pnls / total_pnls
        Profit factor = sum(positive_pnls) / |sum(negative_pnls)|
        Returns infinity if no losses, 0 if no gains and no losses
      ```
   - System Quality Number (SQN)
      ```
          Formula: √(len(pnl_values)) * pnl_mean / pnl_std
          evaluating the system's consistency and trade performance, especially in backtests
          SQN = sqrt(len(pnl_values)) * pnl_mean / pnl_std
          Higher SQN -> more reliable strategy
      ```
 - Plot Results
    1. Portfolio Value
         Trade points:
         Green ^ = Buy
         Red v = Sell
         Yellow o = Hold (sampled, to avoid clutter)
         Blue line - Portfolio's value over time
         Orange line - The asset's price
         Use twinx() to plot price and portfolio value on separate y-axes
     
    2. Cumulative Return 
         Green Line - Cumulative return percentage
         Gray Dashed line - Buy & hold BTC return for comparison
         Shows relative performance between active strategy and passive approach
     



  
