import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
import joypy

# Initialize setup
plt.style.use('seaborn-v0_8')  # Updated seaborn style name
pd.set_option('display.max_columns', 50)

def basic_data_checks(df):
    print("=== Shape ===")
    print(df.shape)
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum().sort_values(ascending=False))
    
    print("\n=== Temporal Consistency ===")
    print("Listings after event date:", 
          df[df['listing_time'] > df['event_date']].shape[0])
    
    print("\n=== Duplicates ===")
    print("Exact duplicates:", df.duplicated().sum())
    
    print("\n=== Numeric Summary ===")
    print(df[['price', 'quantity', 'days_until_event']].describe())
#=== Missing Values ===
#row                 15894
#zone                  210
#section                 7
#
#=== Temporal Consistency ===
#Listings after event date: 17906

#=== Duplicates ===
#Exact duplicates: 351
# Load data and convert dates
#df = pd.read_csv('msg_entertainment_events.csv', parse_dates=['timestamp', 'event_date', 'listing_time'])
#basic_data_checks(df)

#Price Distribution

def plot_price_distributions(df):
    fig, ax = plt.subplots(1, 3, figsize=(18,5))
    
    # Raw price distribution
    sns.histplot(df['price'], bins=50, ax=ax[0], kde=True)
    ax[0].set_title('Raw Price Distribution')
    
    # Log-transformed
    sns.histplot(np.log1p(df['price']), bins=50, ax=ax[1], kde=True) 
    ax[1].set_title('Log-Transformed Prices')
    
    # Price by category
    sns.boxplot(x='Category', y='price', data=df, ax=ax[2])
    ax[2].tick_params(axis='x', rotation=45)
    plt.tight_layout()

def load_data():
    """Load and prepare the dataset"""
    # Read the CSV file without any date parsing
    df = pd.read_csv('etl\msg\msg_standardized_zones.csv')
    
    # Convert date columns with proper handling of ISO format with timezone
    date_columns = ['timestamp', 'event_date', 'listing_time']
    for col in date_columns:
        try:
            # Parse ISO format dates with timezone
            df[col] = pd.to_datetime(df[col], utc=True)
            # Convert to local timezone if needed
            df[col] = df[col].dt.tz_convert('America/New_York')
        except Exception as e:
            print(f"\nError parsing {col}: {str(e)}")
    
    # Print results after conversion
    print("\n=== After Date Conversion ===")
    print("\nSample of converted listing_time values:")
    print(df['listing_time'].head())
    print("\nListing time data type:", df['listing_time'].dtype)
    print("\nNumber of NaT values in listing_time:", df['listing_time'].isna().sum())
    
    return df

def analyze_temporal_patterns(df):
    """
    Analyze how prices change with days until event using filtered data.
    Includes 1-99 percentile filtering, artist-specific analysis, and anomaly detection.
    """
    def create_filtered_df(df):
        """Create filtered dataframes using different methods"""
        # 1-99 percentile filtering
        lower_threshold = df['price'].quantile(0.01)
        upper_threshold = df['price'].quantile(0.99)
        df_99 = df[(df['price'] >= lower_threshold) & (df['price'] <= upper_threshold)].copy()
        
        # Isolation Forest anomaly detection
        df_iso = df.copy()
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        # Reshape price for the model
        price_array = df_iso['price'].values.reshape(-1, 1)
        # Fit and predict
        df_iso['anomaly'] = iso_forest.fit_predict(price_array)
        # Keep only non-anomalies (1)
        df_iso = df_iso[df_iso['anomaly'] == 1].copy()
        df_iso.drop('anomaly', axis=1, inplace=True)
        
        return df_99, df_iso
    
    def analyze_top_artists(df, n_artists=10):
        """Analyze temporal patterns for top artists by number of listings"""
        # Get top artists by number of listings
        top_artists = df['event_name'].value_counts().head(n_artists).index
        
        # Create figure for artist analysis
        plt.figure(figsize=(15, 10))
        
        # Plot for each top artist
        for artist in top_artists:
            artist_data = df[df['event_name'] == artist]
            plt.scatter(artist_data['days_until_event'], artist_data['price'], 
                       alpha=0.5, label=artist)
        
        plt.title('Price vs Days Until Event for Top 10 Artists by Listings')
        plt.xlabel('Days Until Event')
        plt.ylabel('Price ($)')
        plt.yscale('log')  # Log scale for better visualization
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Print statistics for each artist
        print("\n=== Top 10 Artists Analysis ===")
        for artist in top_artists:
            artist_data = df[df['event_name'] == artist]
            print(f"\n{artist}:")
            print(f"Number of listings: {len(artist_data):,}")
            print(f"Average price: ${artist_data['price'].mean():,.2f}")
            print(f"Median price: ${artist_data['price'].median():,.2f}")
            print(f"Price range: ${artist_data['price'].min():,.2f} - ${artist_data['price'].max():,.2f}")
            print(f"Correlation (price vs days): {artist_data['price'].corr(artist_data['days_until_event']):.3f}")
    
    # Create filtered dataframes
    df_99, df_iso = create_filtered_df(df)
    
    # Print summary statistics
    print("\n=== Price Analysis Summary ===")
    
    print("\n1-99 Percentile Filtered Data:")
    print(f"Number of tickets: {len(df_99):,}")
    print(f"Average price: ${df_99['price'].mean():,.2f}")
    print(f"Median price: ${df_99['price'].median():,.2f}")
    print(f"Price range: ${df_99['price'].min():,.2f} - ${df_99['price'].max():,.2f}")
    print(f"Correlation (price vs days): {df_99['price'].corr(df_99['days_until_event']):.3f}")
    
    print("\nIsolation Forest Filtered Data:")
    print(f"Number of tickets: {len(df_iso):,.0f}")
    print(f"Average price: ${df_iso['price'].mean():,.2f}")
    print(f"Median price: ${df_iso['price'].median():,.2f}")
    print(f"Price range: ${df_iso['price'].min():,.2f} - ${df_iso['price'].max():,.2f}")
    print(f"Correlation (price vs days): {df_iso['price'].corr(df_iso['days_until_event']):.3f}")
    
    # Analyze top artists
    analyze_top_artists(df_99)  # Using 1-99 percentile filtered data
    
    return df_99, df_iso

def analyze_seating(df):
    """
    Analyze seating patterns and price distributions across zones, sections, and rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the ticket data with columns: zone, section, row, price
    """
    # Set up the figure with a better layout
    plt.style.use('seaborn-v0_8')  # Updated seaborn style name
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Price Distribution by Zone (Boxplot)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(x='zone', y='price', data=df, ax=ax1)  # Changed from standardized_zone to zone
    ax1.set_title('Price Distribution by Zone', fontsize=12, pad=15)
    ax1.set_xlabel('Zone', fontsize=10)
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Top 10 Most Expensive Sections (Boxplot)
    ax2 = fig.add_subplot(gs[0, 1])
    # Calculate median price for each section
    section_prices = df.groupby('section')['price'].median().sort_values(ascending=False)
    top_sections = section_prices.head(10).index
    sns.boxplot(x='section', y='price', data=df[df['section'].isin(top_sections)], ax=ax2)
    ax2.set_title('Top 10 Most Expensive Sections', fontsize=12, pad=15)
    ax2.set_xlabel('Section', fontsize=10)
    ax2.set_ylabel('Price ($)', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Row Analysis (Scatter plot with trend line)
    ax3 = fig.add_subplot(gs[1, :])
    # Convert row to numeric where possible
    df['row_numeric'] = pd.to_numeric(df['row'], errors='coerce')
    # Remove rows with NaN values for the plot
    valid_rows = df.dropna(subset=['row_numeric'])
    sns.regplot(x='row_numeric', y='price', data=valid_rows.sample(min(10000, len(valid_rows))), 
                scatter_kws={'alpha':0.3}, ax=ax3)
    ax3.set_title('Price vs Row Number', fontsize=12, pad=15)
    ax3.set_xlabel('Row Number', fontsize=10)
    ax3.set_ylabel('Price ($)', fontsize=10)
    
    # 4. Section Popularity (Bar plot)
    ax4 = fig.add_subplot(gs[2, 0])
    section_counts = df['section'].value_counts().head(10)
    sns.barplot(x=section_counts.index, y=section_counts.values, ax=ax4)
    ax4.set_title('Top 10 Most Popular Sections', fontsize=12, pad=15)
    ax4.set_xlabel('Section', fontsize=10)
    ax4.set_ylabel('Number of Listings', fontsize=10)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Price Distribution by Zone (Violin plot)
    ax5 = fig.add_subplot(gs[2, 1])
    sns.violinplot(x='zone', y='price', data=df, ax=ax5)  # Changed from standardized_zone to zone
    ax5.set_title('Price Distribution by Zone (Violin Plot)', fontsize=12, pad=15)
    ax5.set_xlabel('Zone', fontsize=10)
    ax5.set_ylabel('Price ($)', fontsize=10)
    ax5.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Print summary statistics
    print("\n=== Seating Analysis Summary ===")
    print("\nTop 5 Most Expensive Sections:")
    print(section_prices.head().to_frame('Median Price'))
    
    print("\nZone Price Statistics:")
    zone_stats = df.groupby('zone')['price'].agg(['mean', 'median', 'std']).round(2)  # Changed from standardized_zone to zone
    print(zone_stats)
    print("\n Standardized Zone Price Statistics:")
    zone_stats = df.groupby('standardized_zone')['price'].agg(['mean', 'median', 'std']).round(2)  # Changed from standardized_zone to zone
    print(zone_stats)
    '''Mean lot bigger than median for standarized Floor and High std.
    Zone Price Statistics:
                            mean   median        std
    Floor                    7517.77   386.06  837674.32
    '''
    # Clean up temporary column
    df.drop('row_numeric', axis=1, inplace=True)
    
    return fig

def analyze_floor_tickets(df):
    """
    Analyze the top 10 most expensive Floor zone tickets and overall ticket prices.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the ticket data
    """
    # Filter for Floor zone tickets
    floor_tickets = df[df['standardized_zone'] == 'General Admission Floor'].copy()
    
    # Sort by price in descending order and get top 10
    top_floor_tickets = floor_tickets.sort_values('price', ascending=False).head(10)
    
    # Select relevant columns and format the output
    columns_to_show = ['event_name', 'event_date', 'section', 'row', 'price', 
                      'quantity', 'days_until_event', 'listing_time', 'standardized_zone']
    
    # Format the output
    print("\n=== Top 10 Most Expensive Floor Tickets ===")
    print("\nDetailed Information:")
    pd.set_option('display.max_colwidth', None)  # Show full text in columns
    pd.set_option('display.float_format', lambda x: '${:,.2f}'.format(x) if isinstance(x, float) else str(x))
    
    # Display the formatted data
    print(top_floor_tickets[columns_to_show].to_string(index=False))
    
    # Print summary statistics
    print("\nSummary Statistics for Floor Tickets:")
    print(f"Total number of Floor tickets: {len(floor_tickets):,}")
    print(f"Average Floor ticket price: ${floor_tickets['price'].mean():,.2f}")
    print(f"Median Floor ticket price: ${floor_tickets['price'].median():,.2f}")
    print(f"Standard deviation: ${floor_tickets['price'].std():,.2f}")

    
    ''' ChrisStapleton        NaT                 B  15 $99,999,999.99     $2.00               171          NaT
        MattRife        NaT             FLR D   5 $99,999,999.99     $2.00               143          NaT
        ChrisStapleton        NaT                 B  17 $99,999,999.99     $2.00               171          NaT
        SabrinaCarpenter        NaT                 C NaN      $8,112.82     $2.00                 0          NaT
    '''
    # Analyze overall ticket prices
    print("\n=== Overall Ticket Price Analysis ===")
    
    # Get top 20 most expensive tickets
    print("\nTop 20 Most Expensive Tickets Overall:")
    top_20_expensive = df.sort_values('price', ascending=False).head(20)
    print(top_20_expensive[columns_to_show].to_string(index=False))
    
    # Get bottom 20 cheapest tickets
    print("\nTop 20 Cheapest Tickets Overall:")
    top_20_cheapest = df.sort_values('price', ascending=True).head(100)
    print(top_20_cheapest[columns_to_show].to_string(index=False))
    
    # Print overall price statistics
    print("\nOverall Price Statistics:")
    print(f"Total number of tickets: {len(df):,}")
    print(f"Average ticket price: ${df['price'].mean():,.2f}")
    print(f"Median ticket price: ${df['price'].median():,.2f}")
    print(f"Standard deviation: ${df['price'].std():,.2f}")
    print(f"Minimum price: ${df['price'].min():,.2f}")
    print(f"Maximum price: ${df['price'].max():,.2f}")
    
    return top_floor_tickets, top_20_expensive, top_20_cheapest

def analyze_price_statistics(df):
    """
    Analyze price statistics with different filtering methods:
    1. Remove prices over 10000
    2. Remove top 1% of prices
    3. Remove outliers using IQR method per event
    """
    def print_price_stats(df, title):
        """Helper function to print price statistics"""
        print(f"\n=== {title} ===")
        print(f"Number of tickets: {len(df):,.0f}")
        print(f"Average price: ${df['price'].mean():,.2f}")
        print(f"Median price: ${df['price'].median():,.2f}")
        print(f"Standard deviation: ${df['price'].std():,.2f}")
        print(f"Minimum price: ${df['price'].min():,.2f}")
        print(f"Maximum price: ${df['price'].max():,.2f}")
        print(f"25th percentile: ${df['price'].quantile(0.25):,.2f}")
        print(f"75th percentile: ${df['price'].quantile(0.75):,.2f}")
    
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # 1. Remove prices over 10000
    df_clean_10k = df_clean[df_clean['price'] <= 10000].copy()
    print_price_stats(df_clean_10k, "Price Statistics (Prices <= $10,000)")
    
    # 2. Remove top 1% of prices
    price_threshold = df_clean['price'].quantile(0.99)
    df_clean_99 = df_clean[df_clean['price'] <= price_threshold].copy()
    print_price_stats(df_clean_99, "Price Statistics (Removed Top 1%)")
    
    # 3. Remove outliers using IQR method per event
    df_clean_iqr = df_clean.copy()
    
    # Calculate IQR bounds for each event
    event_groups = df_clean_iqr.groupby('event_name')
    filtered_rows = []
    
    for event_name, group in event_groups:
        Q1 = group['price'].quantile(0.25)
        Q3 = group['price'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        filtered_group = group[group['price'] <= upper_bound]
        filtered_rows.append(filtered_group)
    
    df_clean_iqr = pd.concat(filtered_rows)
    print_price_stats(df_clean_iqr, "Price Statistics (IQR Method per Event)")
    
    # Print top 20 rows after IQR filtering
    print("\n=== Top 20 Most Expensive Tickets After IQR Filtering ===")
    columns_to_show = ['event_name', 'event_date', 'section', 'row', 'price', 
                      'quantity', 'days_until_event', 'listing_time', 'standardized_zone']
    
    # Format the output
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda x: '${:,.2f}'.format(x) if isinstance(x, float) else str(x))
    
    top_20_iqr = df_clean_iqr.sort_values('price', ascending=False).head(20)
    print(top_20_iqr[columns_to_show].to_string(index=False))
    
    # Print event-specific statistics
    print("\n=== Event-Specific Price Statistics (After IQR Filtering) ===")
    event_stats = df_clean_iqr.groupby('event_name')['price'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)
    
    # Format the statistics
    event_stats['mean'] = event_stats['mean'].map('${:,.2f}'.format)
    event_stats['median'] = event_stats['median'].map('${:,.2f}'.format)
    event_stats['std'] = event_stats['std'].map('${:,.2f}'.format)
    event_stats['min'] = event_stats['min'].map('${:,.2f}'.format)
    event_stats['max'] = event_stats['max'].map('${:,.2f}'.format)
    
    print(event_stats.sort_values('count', ascending=False))
    
    return df_clean_10k, df_clean_99, df_clean_iqr

def compute_daily_percentiles(df):
    """
    Compute the 10th percentile price for each day and add it as a new column.
    Groups by event_name and event_date. Uses 1-99% filtered data.
    Filters out listings that occur after the event date.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the ticket data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with new column 'daily_10th_percentile' added
    """
    # Create a copy and apply 1-99% filtering
    df_filtered = df.copy()
    lower_threshold = df_filtered['price'].quantile(0.01)
    upper_threshold = df_filtered['price'].quantile(0.99)
    df_filtered = df_filtered[(df_filtered['price'] >= lower_threshold) & 
                            (df_filtered['price'] <= upper_threshold)]
    
    # Calculate days_until_event as event_date - listing_time
    df_filtered['days_until_event'] = (df_filtered['event_date'] - df_filtered['listing_time']).dt.days
    
    # Print listings with negative days (tickets listed after event)
    print("\n=== Listings After Event Date (These will be removed) ===")
    after_event = df_filtered[df_filtered['days_until_event'] < 0].sort_values('days_until_event')
    print("\nSample of listings after event date:")
    print(after_event[['event_name', 'event_date', 'listing_time', 'days_until_event', 'price']].head(10))
    print(f"\nTotal number of listings after event date: {len(after_event):,}")
    
    # Remove listings after event date
    df_filtered = df_filtered[df_filtered['days_until_event'] >= 0]
    print(f"\nRemaining listings after removing post-event listings: {len(df_filtered):,}")
    
    # Save the filtered dataframe to CSV
    output_file = 'etl/msg/msg_filtered_tickets.csv'
    df_filtered.to_csv(output_file, index=False)
    print(f"\nFiltered data saved to: {output_file}")
    
    # Get top 10 artists by number of listings
    top_artists = df_filtered['event_name'].value_counts().head(10).index
    print("\nTop 10 artists by number of listings:")
    for artist in top_artists:
        count = len(df_filtered[df_filtered['event_name'] == artist])
        print(f"{artist}: {count:,} listings")
    
    # Filter for top 10 artists
    df_filtered = df_filtered[df_filtered['event_name'].isin(top_artists)]
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    # First subplot: Hexbin and percentile line
    ax1 = fig.add_subplot(gs[0])
    
    # Create hexbin plot for density
    hexbin = ax1.hexbin(df_filtered['days_until_event'], 
               df_filtered['price'],
               gridsize=50,
               cmap='Blues',
               mincnt=1,
               bins='log',
               zorder=1)
    
    # Group by event_name and event_date
    grouped = df_filtered.groupby(['event_name', 'event_date'])
    
    # Compute 10th percentile for each group
    percentiles = grouped['price'].quantile(0.1).reset_index()
    percentiles.rename(columns={'price': 'daily_10th_percentile'}, inplace=True)
    
    # Calculate days_until_event for the percentiles DataFrame
    percentiles['days_until_event'] = (percentiles['event_date'] - df_filtered['listing_time']).dt.days
    
    # Aggregate and smooth the 10th percentile
    percentiles_sorted = percentiles.sort_values('days_until_event')
    rolling_percentile = percentiles_sorted.set_index('days_until_event')['daily_10th_percentile'].rolling(
        window=7, 
        min_periods=1,
        center=True
    ).mean()
    
    # Plot the smoothed percentile line
    ax1.plot(rolling_percentile.index,
            rolling_percentile.values,
            color='red',
            linewidth=3,
            label='Smoothed 10th Percentile Price',
            zorder=2)
    
    ax1.set_title('Ticket Price Density and 10th Percentile Trend (Top 10 Artists)\nExcluding Post-Event Listings')
    ax1.set_xlabel('Days Until Event from Listing Time')
    ax1.set_ylabel('Price ($)')
    ax1.set_yscale('log')
    plt.colorbar(hexbin, ax=ax1, label='Number of Listings (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: Joyplot
    ax2 = fig.add_subplot(gs[1])
    
    # Create time bins with more reasonable ranges
    df_filtered['time_bin'] = pd.qcut(df_filtered['days_until_event'], 
                                    q=10,  # Use 10 quantiles instead of fixed bins
                                    labels=[f'Bin {i+1}' for i in range(10)])
    
    # Print bin statistics to verify data distribution
    print("\n=== Time Bin Statistics ===")
    bin_stats = df_filtered.groupby('time_bin')['days_until_event'].agg(['count', 'min', 'max'])
    print(bin_stats)
    
    # Create joyplot with simpler parameters
    joypy.joyplot(
        data=df_filtered,
        by='time_bin',
        column='price',
        colormap=plt.cm.viridis,  # Use a different colormap
        overlap=0.5,  # Reduce overlap
        ax=ax2,
        linewidth=1.5
    )
    
    # Set log scale and labels
    ax2.set_xscale('log')
    ax2.set_xlabel('Price ($) - Log Scale')
    ax2.invert_yaxis()
    ax2.set_title('Price Distribution by Time Until Event\n(Top 10 Artists)')
    
    # Add bin information to the plot
    bin_info = df_filtered.groupby('time_bin')['days_until_event'].agg(['min', 'max']).round(0)
    bin_labels = [f'{int(row["min"])}–{int(row["max"])} days' for _, row in bin_info.iterrows()]
    ax2.set_yticklabels(bin_labels)
    
    # Add legend with interpretation
    legend_text = (
        "Distribution Interpretation:\n"
        "• Wider distributions = Higher price variability\n"
        "• Right skew = More expensive tickets\n"
        "• Left skew = More affordable tickets\n"
        "• Multiple peaks = Different price tiers"
    )
    '''
Across every bin—from 0–2 days all the way out to 160–530 days—the skewness is about +1.6 to +2.0 and kurtosis well above 2. That means:

Lots of outliers on the high end.

Mean > median everywhere.

The lowest median is in that 116–159-day window (roughly 4–5 months out). That aligns with where the 10th-pct line bottomed in our earlier plots.

Implication for your model:
Your model should capture a non-monotonic U-shape in price vs. days-out. A simple linear feature for “days to event” will miss that trough. Instead:

Include polynomial terms (e.g. days and days²),

Or fit a spline/GAM,

Or build a piecewise/polynomial regression with a knot around 120 days.


Big stars (Olivia Rodrigo, Harry Styles) naturally start at a much higher “floor” than mid-tier acts.

Implication for your model:
You can’t build one “days-to-price” curve that works for every artist. You need to:

Include artist (or event) as a feature—either as a categorical (one-hot) or with a random effect in a mixed model.

Potentially cluster events into tiers (mega-stars vs. mid-tiers vs. niche acts) and fit separate curves.

    '''
    ax2.text(1.02, 0.5, legend_text,
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             verticalalignment='center')
    
    # Third subplot: Percentile Ribbon Plot
    ax3 = fig.add_subplot(gs[2])
    
    # Compute percentiles for each day
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    pct = df_filtered.groupby('days_until_event')['price'].quantile(qs).unstack()
    
    # Smooth with a short rolling window
    pct_s = pct.rolling(window=7, center=True, min_periods=1).mean()
    
    # Plot ribbons
    ax3.fill_between(pct_s.index, pct_s[0.10], pct_s[0.90],
                    alpha=0.3, color='blue', label='10–90th pct range')
    ax3.fill_between(pct_s.index, pct_s[0.25], pct_s[0.75],
                    alpha=0.5, color='blue', label='25–75th pct range')
    
    # Plot median line
    ax3.plot(pct_s.index, pct_s[0.50], color='red', linewidth=2, label='Median')
    
    # Customize the plot
    ax3.set_yscale('log')
    ax3.set_xlabel('Days Until Event')
    ax3.set_ylabel('Price ($)')
    ax3.set_title('Ticket Price Quantile Ribbons Over Lead Time\n(Top 10 Artists)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Ensure all plots are visible
    plt.subplots_adjust(hspace=0.4)  # Increase space between subplots
    
    # Analyze and print distribution patterns
    print("\n=== Price Distribution Analysis by Time Until Event ===")
    for bin_name, group in df_filtered.groupby('time_bin'):
        days_range = f"{int(group['days_until_event'].min())}–{int(group['days_until_event'].max())} days"
        skew = group['price'].skew()
        kurtosis = group['price'].kurtosis()
        
        print(f"\n{days_range} before event:")
        print(f"• Price range: ${group['price'].min():,.2f} - ${group['price'].max():,.2f}")
        print(f"• Median price: ${group['price'].median():,.2f}")
        print(f"• Mean price: ${group['price'].mean():,.2f}")
        print(f"• Skewness: {skew:.2f} ({'right-skewed' if skew > 0.5 else 'left-skewed' if skew < -0.5 else 'symmetric'})")
        print(f"• Kurtosis: {kurtosis:.2f} ({'heavy-tailed' if kurtosis > 2 else 'light-tailed' if kurtosis < -2 else 'normal'})")
        
        # Additional insights
        if skew > 0.5:
            print("  → Higher prices are more common in this time period")
        elif skew < -0.5:
            print("  → Lower prices are more common in this time period")
        if kurtosis > 2:
            print("  → Price distribution has extreme values")
        elif kurtosis < -2:
            print("  → Price distribution is more uniform")
    
    return df_filtered

if __name__ == "__main__":
    # You can uncomment the section you want to run
    
    # Section 1: Load data
    df = load_data()
    
    # Section 2: Basic data checks
    # basic_data_checks(df)
    
    # Section 3: Price distribution analysis
    # plot_price_distributions(df)
    # plt.show()
    
    # Section 4: Seating Analysis
    # analyze_seating(df)
    # plt.show()
    
    # Section 5: Floor Ticket Analysis
    # analyze_floor_tickets(df)
    
    # Section 6: Price Statistics Analysis
    # analyze_price_statistics(df)
    
    # Section 7: Temporal Price Analysis
    # analyze_temporal_patterns(df)
    # plt.show()
    
    # Section 8: Daily Percentile Analysis
    df_with_percentiles = compute_daily_percentiles(df)
    plt.show()

