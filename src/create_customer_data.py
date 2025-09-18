import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker('en_CA')  # Canadian locale
Faker.seed(42)

def generate_customer_profiles(n_customers=100000, seed=42,):
    """
    Generate 100K customer profiles for trilink in Ontario, Canada
    with logical inter-feature relationships and exact specifications
    """
    
    print(f"Generating {n_customers:,} customer profiles...")
    
    # Initialize customers list - FIX #1
    customers = []
    
    # Ontario postal code data with crime patterns and neighborhood characteristics
    postal_code_data = {
        'Toronto': {
            'weight': 0.45,
            'postal_prefixes': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'],
            'base_median_income': 85000,
            'fiber_availability': 0.85,
            'property_value_multiplier': 1.4,
            # Crime patterns by postal code prefix (crimes per 1000 residents)
            'crime_patterns': {
                'M1': 6.8,   # Scarborough - mixed areas
                'M2': 4.2,   # North York - suburban
                'M3': 5.5,   # North York - mixed
                'M4': 3.8,   # Central - upscale
                'M5': 7.2,   # Downtown - higher density
                'M6': 5.1,   # West Toronto - gentrifying
                'M7': 8.5,   # Etobicoke - some rough areas
                'M8': 6.3,   # Etobicoke - mixed
                'M9': 7.8    # Etobicoke - industrial
            }
        },
        'Mississauga': {
            'weight': 0.18,
            'postal_prefixes': ['L4', 'L5'],
            'base_median_income': 78000,
            'fiber_availability': 0.75,
            'property_value_multiplier': 1.2,
            'crime_patterns': {
                'L4': 4.8,   # Central Mississauga
                'L5': 5.4    # South/West Mississauga
            }
        },
        'Hamilton': {
            'weight': 0.12,
            'postal_prefixes': ['L8', 'L9'],
            'base_median_income': 65000,
            'fiber_availability': 0.65,
            'property_value_multiplier': 0.9,
            'crime_patterns': {
                'L8': 8.2,   # Central Hamilton - older industrial
                'L9': 6.5    # Suburban Hamilton
            }
        },
        'London': {
            'weight': 0.10,
            'postal_prefixes': ['N5', 'N6'],
            'base_median_income': 58000,
            'fiber_availability': 0.60,
            'property_value_multiplier': 0.8,
            'crime_patterns': {
                'N5': 7.1,   # Central/East London
                'N6': 5.8    # West/South London
            }
        },
        'Waterloo': {
            'weight': 0.08,
            'postal_prefixes': ['N2'],
            'base_median_income': 72000,
            'fiber_availability': 0.90,
            'property_value_multiplier': 1.1,
            'crime_patterns': {
                'N2': 4.2    # Generally safe tech hub
            }
        },
        'Guelph': {
            'weight': 0.07,
            'postal_prefixes': ['N1'],
            'base_median_income': 68000,
            'fiber_availability': 0.70,
            'property_value_multiplier': 1.0,
            'crime_patterns': {
                'N1': 3.8    # Generally safe university town
            }
        }
    }
    
    for i in range(n_customers):
        if i % 10000 == 0:
            print(f"Generated {i:,} customers...")
        
        # Generate customer ID in format C00000001 to C00100000
        customer_id = f"C{str(i + 1).zfill(8)}"
        
        # Select city and postal code based on distribution
        city_weights = [data['weight'] for data in postal_code_data.values()]
        city = np.random.choice(list(postal_code_data.keys()), p=city_weights)
        city_info = postal_code_data[city]
        
        # Select postal code prefix within the city
        postal_prefix = np.random.choice(city_info['postal_prefixes'])
        
        # Generate full postal code (Canadian format: A1A 1A1) - FIX #2
        # Canadian postal codes are Letter-Number-Letter Space Number-Letter-Number
        def random_letter():
            return random.choice('ABCDEFGHIJKLMNPRSTUVWXYZ')  # Exclude I, O, Q
        
        def random_digit():
            return str(random.randint(0, 9))
        
        postal_code = f"{postal_prefix[0]}{random_digit()}{random_letter()} {random_digit()}{random_letter()}{random_digit()}"
        
        # Get neighborhood-specific crime rate with some variation
        base_crime_rate = city_info['crime_patterns'][postal_prefix]
        neighborhood_crime_rate = max(1.0, np.random.normal(base_crime_rate, 1.2))
        
        # Calculate neighborhood median income based on crime rate (inverse correlation)
        base_neighborhood_income = city_info['base_median_income']
        # Lower crime areas tend to have higher median income
        crime_income_factor = max(0.6, 1.4 - (neighborhood_crime_rate / 10))
        neighborhood_income_median = max(30000, min(1000000, 
            int(np.random.normal(base_neighborhood_income * crime_income_factor, 15000))))
        
        # === STEP 1: Generate core demographic features ===
        
        # Age (16-100) - normal distribution centered around working age
        age = max(16, min(100, int(np.random.normal(42, 15))))
        
        # === STEP 2: Life stage MUST be consistent with age ===
        if age < 25:
            life_stage = np.random.choice(['Single', 'Young_Family'], p=[0.8, 0.2])
        elif age < 35:
            life_stage = np.random.choice(['Single', 'Young_Family'], p=[0.4, 0.6])
        elif age < 50:
            life_stage = np.random.choice(['Young_Family', 'Established_Family'], p=[0.3, 0.7])
        elif age < 65:
            life_stage = np.random.choice(['Established_Family', 'Empty_Nest'], p=[0.6, 0.4])
        else:  # 65+
            life_stage = np.random.choice(['Empty_Nest', 'Senior'], p=[0.3, 0.7])
        
        # === STEP 3: Family size MUST be consistent with life stage and age ===
        if life_stage == 'Single':
            family_size = np.random.choice([1, 2], p=[0.7, 0.3])  # Single or with partner
        elif life_stage == 'Young_Family':
            family_size = np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3])  # Small families
        elif life_stage == 'Established_Family':
            family_size = np.random.choice([3, 4, 5, 6], p=[0.3, 0.4, 0.2, 0.1])  # Larger families
        elif life_stage == 'Empty_Nest':
            family_size = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])  # Mostly couples
        else:  # Senior
            family_size = np.random.choice([1, 2], p=[0.4, 0.6])  # Singles or couples
        
        # === STEP 4: Education level correlates with age (older = higher chance of advanced degrees) ===
        if city == 'Waterloo':
            # Tech hub - higher education levels
            education_weights = [0.10, 0.25, 0.40, 0.25]  # HS, College, Graduate, Professional
        elif age < 30:
            # Younger people - more likely recent college graduates
            education_weights = [0.20, 0.50, 0.25, 0.05]
        elif age < 50:
            # Peak career - higher advanced degrees
            education_weights = [0.15, 0.35, 0.35, 0.15]
        else:
            # Older generation - more varied education
            education_weights = [0.25, 0.40, 0.25, 0.10]
            
        education_level = np.random.choice(['HS', 'College', 'Graduate', 'Professional'], p=education_weights)
        
        # === STEP 5: Income correlates with age, education, life stage, and location ===
        
        # Base income by age (career progression)
        if age < 25:
            age_income_base = 45000  # Entry level
        elif age < 35:
            age_income_base = 65000  # Early career
        elif age < 50:
            age_income_base = 85000  # Peak earning
        elif age < 65:
            age_income_base = 90000  # Senior roles
        else:
            age_income_base = 55000  # Retirement/pension
        
        # Education multiplier
        education_multipliers = {
            'HS': 0.8,
            'College': 1.0,
            'Graduate': 1.4,
            'Professional': 1.8
        }
        
        # Life stage multiplier (families often have dual income)
        life_stage_multipliers = {
            'Single': 0.8,
            'Young_Family': 1.1,
            'Established_Family': 1.3,
            'Empty_Nest': 1.2,
            'Senior': 0.9
        }
        
        # Location multiplier (neighborhood income affects individual income)
        location_multiplier = neighborhood_income_median / 70000
        
        # Calculate household income
        household_income = max(50000, min(1000000, int(np.random.normal(
            age_income_base * 
            education_multipliers[education_level] * 
            life_stage_multipliers[life_stage] * 
            location_multiplier,
            20000))))
        
        # Income bracket classification
        if household_income < 60000:
            income_bracket = 'Low'
        elif household_income < 90000:
            income_bracket = 'Middle'
        elif household_income < 150000:
            income_bracket = 'Upper_Middle'
        else:
            income_bracket = 'High'
        
        # === STEP 6: Home ownership correlates with age, income, and life stage ===
        ownership_base_prob = 0.3
        
        # Age factor (older = more likely to own)
        if age < 30:
            age_ownership_factor = 0.2
        elif age < 50:
            age_ownership_factor = 0.6
        else:
            age_ownership_factor = 0.8
            
        # Income factor
        income_ownership_factor = min(0.4, household_income / 200000)
        
        # Life stage factor
        life_stage_ownership_factors = {
            'Single': 0.1,
            'Young_Family': 0.3,
            'Established_Family': 0.4,
            'Empty_Nest': 0.3,
            'Senior': 0.2
        }
        
        ownership_prob = min(0.95, ownership_base_prob + age_ownership_factor + 
                           income_ownership_factor + life_stage_ownership_factors[life_stage])
        home_ownership = np.random.choice(['Own', 'Rent', 'Other'], 
                                        p=[ownership_prob, (1-ownership_prob)*0.9, (1-ownership_prob)*0.1])
        
        # === STEP 7: Home type correlates with income, family size, and ownership ===
        if income_bracket == 'High' and family_size >= 3:
            home_type_probs = [0.7, 0.2, 0.05, 0.05]  # Single_Family, Townhouse, Condo, Apartment
        elif income_bracket in ['Upper_Middle', 'High']:
            home_type_probs = [0.5, 0.3, 0.15, 0.05]
        elif income_bracket == 'Middle':
            home_type_probs = [0.3, 0.3, 0.25, 0.15]
        else:  # Low income
            home_type_probs = [0.1, 0.2, 0.3, 0.4]
            
        # Adjust for family size - FIX #3: Proper probability normalization
        if family_size >= 4:
            # Large families need more space
            home_type_probs[0] += 0.2  # More single family
            home_type_probs[3] = max(0.01, home_type_probs[3] - 0.2)  # Less apartments (but not negative)
        elif family_size == 1:
            # Singles prefer condos/apartments
            home_type_probs[0] = max(0.01, home_type_probs[0] - 0.2)  # Less single family
            home_type_probs[2] += 0.1  # More condos
            home_type_probs[3] += 0.1  # More apartments
            
        # Ensure all probabilities are positive and normalize to sum to 1.0
        home_type_probs = [max(0.01, p) for p in home_type_probs]
        total_prob = sum(home_type_probs)
        home_type_probs = [p/total_prob for p in home_type_probs]
        
        home_type = np.random.choice(['Single_Family', 'Townhouse', 'Condo', 'Apartment'], 
                                   p=home_type_probs)
        
        # === STEP 8: Property characteristics correlate with home type and income ===
        
        # Square footage based on home type and family size
        sqft_base = {
            'Single_Family': 2000,
            'Townhouse': 1300,
            'Condo': 900,
            'Apartment': 700
        }
        
        family_size_factor = 1 + (family_size - 2) * 0.15  # More space for larger families
        income_size_factor = household_income / 80000  # Higher income = larger homes
        
        home_square_footage = max(500, min(5000, int(np.random.normal(
            sqft_base[home_type] * family_size_factor * income_size_factor, 300))))
        
        # Property value based on location, home type, size, and crime rate
        base_property_value = city_info['property_value_multiplier'] * 400000
        
        # Crime affects property values (inverse relationship)
        crime_property_factor = max(0.5, 1.3 - (neighborhood_crime_rate / 12))
        
        # Home type multipliers
        home_type_multipliers = {
            'Single_Family': 1.3,
            'Townhouse': 1.0,
            'Condo': 0.8,
            'Apartment': 0.6
        }
        
        # Size factor
        size_factor = home_square_footage / 1500
        
        property_value = max(50000, min(2000000, int(np.random.normal(
            base_property_value * 
            crime_property_factor * 
            home_type_multipliers[home_type] * 
            size_factor,
            100000))))
        
        # === STEP 9: Work from home correlates with education, income, age, and city ===
        wfh_base_prob = 0.1
        
        if city == 'Waterloo':
            wfh_base_prob = 0.2  # Tech hub
        
        if education_level in ['Graduate', 'Professional']:
            wfh_base_prob += 0.15
        
        if income_bracket in ['Upper_Middle', 'High']:
            wfh_base_prob += 0.10
            
        if age < 35:
            wfh_base_prob += 0.05  # Younger workers more remote
        elif age > 55:
            wfh_base_prob -= 0.05  # Older workers more traditional
            
        work_from_home_flag = np.random.random() < wfh_base_prob
        
        # === STEP 10: Fiber availability correlates with income and location ===
        fiber_base_prob = city_info['fiber_availability']
        # Higher income neighborhoods get fiber first
        income_fiber_boost = max(0, (neighborhood_income_median - 60000) / 100000 * 0.15)
        fiber_availability = (fiber_base_prob + income_fiber_boost) > np.random.random()
        
        # Create customer record with exact feature specifications
        customer = {
            'customer_id': customer_id,
            'city': city,
            'postal_code': postal_code,
            
            # 1. Demographics & Household (7 features)
            'age': age,
            'household_income': household_income,
            'income_bracket': income_bracket,
            'family_size': family_size,
            'home_ownership': home_ownership,
            'work_from_home_flag': work_from_home_flag,
            'education_level': education_level,
            'life_stage': life_stage,
            
            # 2. Property & Location (6 features)
            'home_type': home_type,
            'home_square_footage': home_square_footage,
            'property_value': property_value,
            'neighborhood_crime_rate': round(neighborhood_crime_rate, 1),
            'neighborhood_income_median': neighborhood_income_median,
            'fiber_availability': fiber_availability,
            
            'created_timestamp': datetime.now()
        }
        
        customers.append(customer)
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = pd.DataFrame(customers)
    
    # Display summary statistics and data quality checks
    print(f"\n📊 Generated {len(df):,} customer profiles")
    print(f"🏙️  Cities: {df['city'].value_counts().to_dict()}")
    print(f"💰 Income brackets: {df['income_bracket'].value_counts().to_dict()}")
    print(f"🏠 Home types: {df['home_type'].value_counts().to_dict()}")
    print(f"👨‍👩‍👧‍👦 Life stages: {df['life_stage'].value_counts().to_dict()}")
    print(f"📚 Education: {df['education_level'].value_counts().to_dict()}")
    print(f"📶 Fiber availability: {df['fiber_availability'].sum():,} customers ({df['fiber_availability'].mean():.1%})")
    print(f"🏠 Work from home: {df['work_from_home_flag'].sum():,} customers ({df['work_from_home_flag'].mean():.1%})")
    print(f"🏡 Home ownership: {df['home_ownership'].value_counts().to_dict()}")
    
    # Data quality checks
    print(f"\n🔍 Data Quality Checks:")
    print(f"Age range: {df['age'].min()}-{df['age'].max()}")
    print(f"Income range: ${df['household_income'].min():,.0f}-${df['household_income'].max():,.0f}")
    print(f"Property value range: ${df['property_value'].min():,.0f}-${df['property_value'].max():,.0f}")
    print(f"Square footage range: {df['home_square_footage'].min()}-{df['home_square_footage'].max()}")
    print(f"Family size range: {df['family_size'].min()}-{df['family_size'].max()}")
    print(f"Crime rate range: {df['neighborhood_crime_rate'].min():.1f}-{df['neighborhood_crime_rate'].max():.1f}")
    
    # Check logical consistency
    print(f"\n✅ Logic Consistency Checks:")
    seniors_under_30 = len(df[(df['age'] < 30) & (df['life_stage'] == 'Senior')])
    young_families_over_60 = len(df[(df['age'] > 60) & (df['life_stage'] == 'Young_Family')])
    large_families_singles = len(df[(df['family_size'] > 4) & (df['life_stage'] == 'Single')])
    
    print(f"Seniors under 30: {seniors_under_30} (should be 0)")
    print(f"Young families over 60: {young_families_over_60} (should be 0)")
    print(f"Large families marked as Single: {large_families_singles} (should be 0)")
    
    # Sample postal codes to verify format
    print(f"\n📮 Sample postal codes: {df['postal_code'].head(10).tolist()}")
    
    return df

# Generate the dataset
if __name__ == "__main__":
    customers_df = generate_customer_profiles(10)
    
    # Save to CSV
    filename = f".//data//trilink_customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    customers_df.to_csv(filename, index=False)
    print(f"\n💾 Saved to: {filename}")
    
