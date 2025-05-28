import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import MEstimateEncoder
import numpy as np

# Define section categories based on naming patterns
section_categories = {
    'VIP': r'VIP|Premium|Club|Lounge|Madison Club|The Lounges',
    'GA': r'GA|General Admission|Floor|Pit|Standing|FLOOR|FLR [A-H]',
    'Wheelchair': r'Wheelchair|Accessible|WC|LWC|UWC',
    'Level': r'\bLevel\b|\bL\d{3}\b|UPPER|LOWER BOWL|upper \d{3}|Upper \d{3}',
    'Suite': r'Suite|ELS|Event Level',
    'Barstool': r'Barstool|Bar Stool',
    'Chase': r'Chase',
    'Numerical': r'^\d{3}$|^\d{2}$|^\d{1,3}[PF]?$|^\d{1,3}\s*[PF]$|^\d{1,3}\s*Side\s*(?:Stage|View)$',
    'Riser': r'Riser|RS|SS',
    'Special': r'MC\s*\d+|LS\s*\d+|TBD|SRO|SEC\s*[A-Z]|SECTION\s*[A-Z]|s\s*\d{3}',
    'Letter': r'^[A-M]$'  # New category for single letter sections
}

def clean_section(section):
    """Clean and standardize section names"""
    if pd.isna(section):
        return 'Unknown'
    
    # Convert to string and strip whitespace
    section = str(section).strip()
    
    # Remove extra spaces
    section = re.sub(r'\s+', ' ', section)
    
    # Standardize common variations
    section = section.replace('SECTION', 'SEC')
    section = section.replace('Section', 'SEC')
    
    # Remove 'The' from section names
    section = re.sub(r'^The\s+', '', section, flags=re.IGNORECASE)
    
    # Remove dashes and clean up spaces
    section = re.sub(r'^-\s*', '', section)  # Remove leading dash
    section = re.sub(r'\s*-\s*', ' ', section)  # Replace internal dashes with space
    
    return section

def categorize_section(section):
    """Categorize a section based on its name"""
    section = clean_section(section)
    
    if section == 'Unknown':
        return 'Unknown'
    
    # Check against all patterns
    for category, pattern in section_categories.items():
        if re.search(pattern, section, flags=re.IGNORECASE):
            return category
    
    return 'Other'

def analyze_sections():
    # Load the data
    print("Loading data...")
    df = pd.read_csv("etl/msg/msg_filtered_tickets_cleaned.csv")
    
    # Remove parking entries
    df = df[~df['section'].str.contains('Parking', case=False, na=False)]
    
    # Apply categorization
    print("\nCategorizing sections...")
    df['section_category'] = df['section'].apply(categorize_section)
    
    # Get section counts
    section_counts = df['section_category'].value_counts()
    
    # Print results
    print("\nSection Categories and Counts:")
    print("-" * 40)
    for category, count in section_counts.items():
        print(f"{category:<15} {count:>10,}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    section_counts.plot(kind='bar')
    plt.title('Distribution of Section Categories')
    plt.xlabel('Section Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('section_distribution.png')
    print("\nSaved section distribution plot to 'section_distribution.png'")
    
    # Print some example sections for each category
    print("\nExample sections for each category:")
    print("-" * 40)
    for category in section_counts.index:
        examples = df[df['section_category'] == category]['section'].unique()[:5]
        print(f"\n{category}:")
        for example in examples:
            print(f"  - {example}")
    
    # Analyze letter sections
    print("\nLetter Sections Analysis:")
    print("-" * 40)
    letter_sections = df[df['section_category'] == 'Letter']
    
    # Count by letter
    print("\nCount by Letter:")
    letter_counts = letter_sections['section'].value_counts()
    for letter, count in letter_counts.items():
        print(f"{letter:<5} {count:>10,}")
    
    # Count by event
    print("\nEvents using Letter Sections:")
    print("-" * 40)
    event_counts = letter_sections['event_name'].value_counts()
    for event, count in event_counts.items():
        print(f"{event:<50} {count:>10,}")
    
    # Analyze numerical sections
    print("\nNumerical Sections Analysis:")
    print("-" * 40)
    numerical_sections = df[df['section_category'] == 'Numerical']
    
    # Count by section number
    print("\nCount by Section Number:")
    numerical_counts = numerical_sections['section'].value_counts()
    
    # Group by section number pattern
    section_patterns = {
        '100s': r'^1\d{2}$',
        '200s': r'^2\d{2}$',
        '300s': r'^3\d{2}$',
        '400s': r'^4\d{2}$',
        'Single/Double Digits': r'^\d{1,2}$',
        'With P/F': r'\d+[PF]',
        'With Side View/Stage': r'\d+\s*Side\s*(?:Stage|View)'
    }
    
    print("\nCount by Section Pattern:")
    print("-" * 40)
    for pattern_name, pattern in section_patterns.items():
        count = numerical_sections[numerical_sections['section'].str.match(pattern, na=False)].shape[0]
        print(f"{pattern_name:<20} {count:>10,}")
    
    print("\nTop 20 Most Common Numerical Sections:")
    print("-" * 40)
    for section, count in numerical_counts.head(20).items():
        print(f"{section:<10} {count:>10,}")
    
    # Print remaining sections in Other category with their counts
    print("\nRemaining sections in 'Other' category with counts:")
    print("-" * 40)
    other_sections = df[df['section_category'] == 'Other']['section'].value_counts()
    for section, count in other_sections.items():
        print(f"{section:<30} {count:>10,}")
    
    # Apply M-Estimate encoding
    print("\nApplying M-Estimate encoding to section categories...")
    encoder = MEstimateEncoder(cols=['section_category'], m=50)
    df['section_encoded'] = encoder.fit_transform(df['section_category'], df['price'])
    
    # Analyze encoded values
    print("\nSection Category Price Analysis (M-Estimate Encoded):")
    print("-" * 40)
    encoded_analysis = df.groupby('section_category').agg({
        'section_encoded': 'mean',
        'price': ['mean', 'std', 'count']
    }).round(2)
    
    # Sort by encoded value
    encoded_analysis = encoded_analysis.sort_values(('section_encoded', 'mean'), ascending=False)
    
    # Print analysis
    print("\nSection Categories by Average Encoded Price:")
    print("-" * 80)
    print(f"{'Category':<15} {'Encoded Value':<15} {'Mean Price':<15} {'Std Dev':<15} {'Count':<10}")
    print("-" * 80)
    for idx, row in encoded_analysis.iterrows():
        print(f"{idx:<15} {row[('section_encoded', 'mean')]:<15.2f} {row[('price', 'mean')]:<15.2f} {row[('price', 'std')]:<15.2f} {row[('price', 'count')]:<10,}")
    
    # Create visualization of encoded values
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='section_category', y='price')
    plt.title('Price Distribution by Section Category')
    plt.xlabel('Section Category')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('section_price_distribution.png')
    print("\nSaved section price distribution plot to 'section_price_distribution.png'")

    # Save the encoded DataFrame
    print("\nSaving encoded DataFrame to CSV...")
    df.to_csv('MSG_Sections_Encoded.csv', index=False)
    print("Saved encoded DataFrame to 'MSG_Sections_Encoded.csv'")

if __name__ == "__main__":
    analyze_sections() 