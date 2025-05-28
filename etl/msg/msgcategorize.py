import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def categorize_event(event_name):
    sports_keywords = ['StJohn', 'NewYorkKnicks', 'NewYorkRangers', 'CBSSportsClassic', 
                      'JimmyVClassic', 'ChampionsClassic', 'UFC2', 'ProBullRiding', 
                      'Boxing', 'Basketball', 'Hockey', 'NationalInvitationalTournament', 'PBR', 'NCAA', 'Knicks', 'Rangers', 'WWE', 'TaylorvsSerrano']
    comedy_keywords = ['SebastianManiscalco', 'AndrewSchulz', 'NewYorkComedyFestivalBillBurr',
                      'NateBargatze', 'JohnMulaney', 'DaveChappelleLive', 'DaveChappelle',
                      'ChrisRockKevinHart', 'LouisCK', 'MattRife', 'TomSegura',
                      'JohnMulaneyFromScratch', 'KILLTONYLIVEFROMMADISONSQUAREGARDEN',
                      'NYCStillRisingAfter20YearsAComedyCelebrationDaveAttellBillBurrDaveChappelle',
                      'JoeRogan', 'JoKoy', 'DropoutLiveNationPresentDimension20GauntletatTheGarden',
                      'KillTony', 'NickCannonsWildNOutLive', 'GardenofLaughs', 'TrevorNoah']
    other_keywords = ['MadisonSquareGardenAllAccessTour', 'Parking', 'MadisonSquareGardenTourExperience', 'WestminsterKennelClubDogShow']
    
    if any(keyword in event_name for keyword in sports_keywords):
        return 'sports'
    elif any(keyword in event_name for keyword in comedy_keywords):
        return 'comedy'
    elif any(keyword in event_name for keyword in other_keywords):
        return 'other'
    return 'Music'

def analyze_prices_by_category(df):
    """
    Analyze average ticket prices by category, zone and section
    """
    logger.info("Calculating price statistics...")
    
    # Group by category, zone and section and calculate mean price
    price_analysis = df.groupby(['Category', 'zone', 'section'])['price'].agg([
        'mean',
        'count'
    ]).round(2).reset_index()
    
    # Sort by category and mean price descending
    price_analysis = price_analysis.sort_values(['Category', 'mean'], ascending=[True, False])
    
    # Print results by category
    for category in price_analysis['Category'].unique():
        print(f"\n=== {category.upper()} EVENTS PRICE ANALYSIS ===")
        
        # Print zone summaries
        zone_summary = df[df['Category'] == category].groupby('zone')['price'].agg([
            'mean',
            'min',
            'max',
            'count'
        ]).round(2)
        
        print(f"\nZone Summary for {category.upper()} Events:")
        print(zone_summary)
        
        # Print top 3 highest priced listings for each zone
        print(f"\nTop 3 Highest Priced Listings by Zone for {category.upper()} Events:")
        category_df = df[df['Category'] == category]
        for zone in category_df['zone'].unique():
            print(f"\nZone: {zone}")
            zone_listings = category_df[category_df['zone'] == zone].sort_values('price', ascending=False).head(3)
            for _, row in zone_listings.iterrows():
                print(f"Event: {row['event_name']}")
                print(f"Section: {row['section']}")
                print(f"Price: ${row['price']:.2f}")
                print("-" * 30)
        
        # Print total events in category
        event_count = len(df[df['Category'] == category]['event_name'].unique())
        print(f"\nTotal unique events in {category.upper()}: {event_count}")
        print("=" * 50)

try:
    # Load MSG data
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "msg_with_features.csv")
    logger.info(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Add Category column
    df['Category'] = df['event_name'].apply(categorize_event)
    
    # Print category distribution
    print("\nCategory Distribution:")
    print("=" * 50)
    print(df['Category'].value_counts())
    
    # Analyze prices by category
    #Seats Could be different for each event based on stage setup
    print("\nPRICE ANALYSIS BY CATEGORY")
    print("=" * 50)
    analyze_prices_by_category(df)
    
    # Filter for Music and Comedy events and save to CSV
    entertainment_df = df[df['Category'].isin(['Music', 'comedy'])]
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "msg_entertainment_events.csv")
    entertainment_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(entertainment_df)} entertainment events to {output_path}")
    
except Exception as e:
    logger.error(f"Error processing MSG data: {str(e)}")
    raise
