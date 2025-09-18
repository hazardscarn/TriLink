import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_internet_services_fixed(customers_df):
    """
    Generate internet services with fixed contract logic
    """
    print("Generating internet services with fixed contract logic...")
    
    internet_services = []
    service_counter = 1
    current_date = datetime.now()
    
    for _, customer in customers_df.iterrows():
        # Internet service adoption probability (same as before)
        adoption_prob = 0.4
        
        # Apply same adoption factors as before
        if 25 <= customer['age'] <= 55:
            adoption_prob += 0.2
        elif customer['age'] < 25:
            adoption_prob += 0.1
        elif customer['age'] > 65:
            adoption_prob -= 0.1
            
        if customer['income_bracket'] in ['Upper_Middle', 'High']:
            adoption_prob += 0.15
        elif customer['income_bracket'] == 'Low':
            adoption_prob -= 0.1
            
        if customer['work_from_home_flag']:
            adoption_prob += 0.25
            
        if customer['family_size'] >= 4:
            adoption_prob += 0.1
        elif customer['family_size'] == 1:
            adoption_prob -= 0.05
            
        if customer['fiber_availability']:
            adoption_prob += 0.1
            
        if customer['home_type'] == 'Apartment':
            adoption_prob -= 0.15
        elif customer['home_type'] == 'Single_Family':
            adoption_prob += 0.05
            
        if np.random.random() > adoption_prob:
            continue
            
        # Generate initial service start date
        start_date = current_date - timedelta(days=np.random.randint(30, 1095))
        
        # Contract type selection
        if customer['income_bracket'] == 'High':
            contract_probs = [0.4, 0.35, 0.25]
        elif customer['income_bracket'] == 'Low':
            contract_probs = [0.2, 0.4, 0.4]
        else:
            contract_probs = [0.35, 0.4, 0.25]
            
        contract_type = np.random.choice(['Month_to_Month', '12_Month', '24_Month'], p=contract_probs)
        
        # Plan tier selection
        plan_tier_probs = [0.3, 0.4, 0.3]
        
        if customer['income_bracket'] == 'High':
            plan_tier_probs = [0.1, 0.3, 0.6]
        elif customer['income_bracket'] == 'Low':
            plan_tier_probs = [0.6, 0.3, 0.1]
            
        if customer['work_from_home_flag']:
            plan_tier_probs[2] += 0.2
            plan_tier_probs[0] = max(0.05, plan_tier_probs[0] - 0.2)
            
        if customer['family_size'] >= 4:
            plan_tier_probs[1] += 0.1
            plan_tier_probs[0] = max(0.05, plan_tier_probs[0] - 0.1)
            
        if not customer['fiber_availability']:
            plan_tier_probs[2] = 0.05
            plan_tier_probs[0] += 0.25
            
        plan_tier_probs = [max(0.05, p) for p in plan_tier_probs]
        total_prob = sum(plan_tier_probs)
        plan_tier_probs = [p/total_prob for p in plan_tier_probs]
        
        plan_tier = np.random.choice(['Basic_25', 'Standard_100', 'Premium_Gig'], p=plan_tier_probs)
        
        # Set speed and cost
        if plan_tier == 'Basic_25':
            speed_mbps = 25
            monthly_cost = 39 + np.random.randint(-5, 10)
        elif plan_tier == 'Standard_100':
            speed_mbps = 100
            monthly_cost = 69 + np.random.randint(-5, 10)
        else:
            speed_mbps = 1000
            monthly_cost = 99 + np.random.randint(-5, 15)
        
        # SIMPLIFIED CONTRACT LOGIC
        # Calculate original contract end date
        if contract_type == '12_Month':
            original_contract_end = start_date + relativedelta(months=12)
        elif contract_type == '24_Month':
            original_contract_end = start_date + relativedelta(months=24)
        else:
            original_contract_end = None
        
        # For active customers, ensure contract end date makes sense
        current_contract_type = contract_type
        current_contract_end = original_contract_end
        
        # If original contract would have ended in the past, simulate renewal/conversion
        if original_contract_end and original_contract_end < current_date:
            if np.random.random() < 0.7:  # 70% renew with new contract
                # Create new contract starting from original end date
                if contract_type == '24_Month':
                    new_contract_type = np.random.choice(['12_Month', '24_Month'], p=[0.4, 0.6])
                else:
                    new_contract_type = np.random.choice(['12_Month', '24_Month'], p=[0.6, 0.4])
                
                # Calculate new end date
                if new_contract_type == '12_Month':
                    current_contract_end = original_contract_end + relativedelta(months=12)
                else:
                    current_contract_end = original_contract_end + relativedelta(months=24)
                    
                current_contract_type = new_contract_type
                
                # If still in the past, add another renewal
                while current_contract_end < current_date:
                    if current_contract_type == '12_Month':
                        current_contract_end += relativedelta(months=12)
                    else:
                        current_contract_end += relativedelta(months=24)
            else:
                # Convert to month-to-month
                current_contract_type = 'Month_to_Month'
                current_contract_end = None
        
        # Churn logic
        churn_prob = 0.15
        
        # Adjust churn probability
        if plan_tier == 'Basic_25' and customer['work_from_home_flag']:
            churn_prob += 0.25
        if not customer['fiber_availability'] and plan_tier != 'Basic_25':
            churn_prob += 0.15
        if customer['income_bracket'] == 'Low':
            churn_prob += 0.10
        if customer['age'] > 65:
            churn_prob -= 0.05
        
        # Contract affects churn
        if current_contract_type in ['12_Month', '24_Month'] and current_contract_end and current_contract_end > current_date:
            churn_prob *= 0.3  # Much lower churn during active contract
        
        # Calculate if customer churned
        service_days = (current_date - start_date).days
        annual_churn_prob = churn_prob * (service_days / 365)
        
        if np.random.random() < annual_churn_prob:
            # Customer churned
            max_churn_days = service_days
            if current_contract_end and current_contract_end < current_date:
                # If contract ended, churn could be anytime after contract end
                days_after_contract = (current_date - current_contract_end).days
                churn_days_from_start = (current_contract_end - start_date).days + np.random.randint(0, max(1, days_after_contract))
            else:
                # Churn during contract or month-to-month
                churn_days_from_start = np.random.randint(30, max_churn_days + 1)
            
            end_date = start_date + timedelta(days=churn_days_from_start)
        else:
            end_date = None
        
        # Calculate other metrics
        base_usage = speed_mbps * 0.3
        
        if customer['work_from_home_flag']:
            usage_multiplier = 2.0
        elif customer['family_size'] >= 4:
            usage_multiplier = 1.5
        elif customer['family_size'] == 1:
            usage_multiplier = 0.6
        else:
            usage_multiplier = 1.0
            
        data_usage_gb = max(50, int(np.random.normal(base_usage * usage_multiplier, 50)))
        
        base_devices = 5 + customer['family_size'] * 2
        if customer['income_bracket'] in ['Upper_Middle', 'High']:
            device_multiplier = 1.4
        else:
            device_multiplier = 1.0
            
        connected_devices = max(2, int(np.random.normal(base_devices * device_multiplier, 3)))
        
        # Complaints and outages
        if end_date is not None:
            speed_complaints = np.random.poisson(3.5)
            outage_count = np.random.poisson(2.8)
        else:
            if plan_tier == 'Basic_25':
                speed_complaints = np.random.poisson(1.5)
                outage_count = np.random.poisson(1.2)
            elif plan_tier == 'Standard_100':
                speed_complaints = np.random.poisson(0.8)
                outage_count = np.random.poisson(0.9)
            else:
                speed_complaints = np.random.poisson(0.4)
                outage_count = np.random.poisson(0.6)
                
        if customer['work_from_home_flag']:
            speed_complaints = int(speed_complaints * 1.5)
            outage_count = int(outage_count * 1.3)
        
        # Early termination flag
        early_termination = 0
        if end_date and current_contract_end and end_date < current_contract_end:
            early_termination = 1
            
        service = {
            'service_id': f"INT_{str(service_counter).zfill(8)}",
            'customer_id': customer['customer_id'],
            'plan_tier': plan_tier,
            'speed_mbps': speed_mbps,
            'monthly_cost': monthly_cost,
            'data_usage_gb': data_usage_gb,
            'connected_devices': connected_devices,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d') if end_date else None,
            'contract_type': current_contract_type,
            'contract_end_date': current_contract_end.strftime('%Y-%m-%d') if current_contract_end else None,
            'speed_complaints': speed_complaints,
            'outage_count': outage_count,
            'internet_churn': 1 if end_date else 0,
            'early_termination': early_termination
        }
        
        internet_services.append(service)
        service_counter += 1
        
    df = pd.DataFrame(internet_services)
    
    # Add calculated fields
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['contract_end_date'] = pd.to_datetime(df['contract_end_date'])
    
    # Calculate tenure
    df['internet_tenure_days'] = np.where(
        df['end_date'].isnull(),
        (pd.to_datetime('today') - df['start_date']).dt.days,
        (df['end_date'] - df['start_date']).dt.days
    )
    
    # Contract completion percentage
    df['contract_completed_percent'] = np.where(
        df['contract_type'] == 'Month_to_Month',
        np.nan,
        np.where(
            df['end_date'].isnull(),
            np.minimum(
                ((pd.to_datetime('today') - df['start_date']).dt.days /
                 (df['contract_end_date'] - df['start_date']).dt.days) * 100,
                100
            ),
            np.minimum(
                ((df['end_date'] - df['start_date']).dt.days /
                 (df['contract_end_date'] - df['start_date']).dt.days) * 100,
                100
            )
        )
    )
    
    print(f"Generated {len(df):,} internet services")
    print(f"Active services: {len(df[df['end_date'].isna()]):,}")
    print(f"Churned services: {len(df[df['end_date'].notna()]):,}")
    print(f"Early terminations: {df['early_termination'].sum()}")
    print(f"Contract distribution: {df['contract_type'].value_counts().to_dict()}")
    
    # Validation: Check that active customers don't have past contract end dates
    active_df = df[df['end_date'].isna()].copy()
    active_with_contracts = active_df[active_df['contract_end_date'].notna()]
    
    if len(active_with_contracts) > 0:
        past_contracts = active_with_contracts[active_with_contracts['contract_end_date'] < pd.to_datetime('today')]
        print(f"Validation: Active customers with past contract end dates: {len(past_contracts)}")
        if len(past_contracts) > 0:
            print("⚠️  Warning: Some active customers have past contract end dates")
        else:
            print("✅ Validation passed: No active customers with past contract end dates")
    
    return df

def generate_mobile_services_fixed(customers_df):
    """
    Generate mobile services with fixed contract logic
    """
    print("\nGenerating mobile services with fixed contract logic...")
    
    mobile_services = []
    service_counter = 1
    current_date = datetime.now()
    
    for _, customer in customers_df.iterrows():
        # Mobile adoption logic
        adoption_prob = 0.6
        
        if customer['age'] < 35:
            adoption_prob += 0.15
        elif customer['age'] > 70:
            adoption_prob -= 0.10
            
        if customer['income_bracket'] == 'Low':
            adoption_prob -= 0.05
        elif customer['income_bracket'] == 'High':
            adoption_prob += 0.10
            
        if customer['family_size'] >= 3:
            adoption_prob += 0.15
            
        if np.random.random() > adoption_prob:
            continue
            
        # Generate start date
        start_date = current_date - timedelta(days=np.random.randint(60, 1095))
        
        # Line count logic
        if customer['life_stage'] == 'Single':
            line_count = 1
        elif customer['life_stage'] in ['Young_Family', 'Established_Family']:
            if customer['family_size'] <= 2:
                line_count = np.random.choice([1, 2], p=[0.3, 0.7])
            elif customer['family_size'] <= 4:
                line_count = np.random.choice([2, 3, 4], p=[0.2, 0.4, 0.4])
            else:
                line_count = min(customer['family_size'], np.random.choice([4, 5, 6], p=[0.4, 0.4, 0.2]))
        else:
            line_count = np.random.choice([1, 2], p=[0.4, 0.6])
            
        family_plan_flag = line_count > 1
        
        # Plan type selection
        if customer['income_bracket'] == 'High':
            plan_type = np.random.choice(['Unlimited_Premium', 'Unlimited_Standard', 'Limited_10GB'], p=[0.5, 0.4, 0.1])
        elif customer['income_bracket'] == 'Upper_Middle':
            plan_type = np.random.choice(['Unlimited_Premium', 'Unlimited_Standard', 'Limited_10GB'], p=[0.2, 0.6, 0.2])
        elif customer['income_bracket'] == 'Middle':
            plan_type = np.random.choice(['Unlimited_Standard', 'Limited_10GB', 'Limited_5GB'], p=[0.4, 0.4, 0.2])
        else:
            plan_type = np.random.choice(['Limited_10GB', 'Limited_5GB', 'Limited_2GB'], p=[0.3, 0.4, 0.3])
        
        # Contract type for mobile
        if family_plan_flag and customer['income_bracket'] in ['Upper_Middle', 'High']:
            contract_probs = [0.2, 0.4, 0.3, 0.1]  # Include 36-month option
            contract_options = ['Month_to_Month', '12_Month', '24_Month', '36_Month']
        elif customer['age'] < 30:
            contract_probs = [0.6, 0.3, 0.1]
            contract_options = ['Month_to_Month', '12_Month', '24_Month']
        else:
            contract_probs = [0.4, 0.4, 0.2]
            contract_options = ['Month_to_Month', '12_Month', '24_Month']
        
        contract_type = np.random.choice(contract_options, p=contract_probs)
        
        # Calculate contract end date
        if contract_type == '12_Month':
            contract_end_date = start_date + relativedelta(months=12)
        elif contract_type == '24_Month':
            contract_end_date = start_date + relativedelta(months=24)
        elif contract_type == '36_Month':
            contract_end_date = start_date + relativedelta(months=36)
        else:
            contract_end_date = None
        
        # Handle contract renewals for active customers
        if contract_end_date and contract_end_date < current_date:
            if np.random.random() < 0.7:  # 70% renew
                # Extend contract
                months_to_add = 12 if contract_type in ['12_Month', '36_Month'] else 24
                while contract_end_date < current_date:
                    contract_end_date += relativedelta(months=months_to_add)
            else:
                # Convert to month-to-month
                contract_type = 'Month_to_Month'
                contract_end_date = None
        
        # Simplified churn logic
        churn_prob = 0.20
        if 'Unlimited' in plan_type:
            churn_prob -= 0.05
        if family_plan_flag:
            churn_prob -= 0.08
        if customer['income_bracket'] == 'Low':
            churn_prob += 0.15
        
        service_days = (current_date - start_date).days
        annual_churn_prob = churn_prob * (service_days / 365)
        
        if np.random.random() < annual_churn_prob:
            min_service_days = max(1, min(90, service_days))
            churn_days_after_start = np.random.randint(min_service_days, service_days + 1)
            end_date = start_date + timedelta(days=churn_days_after_start)
        else:
            end_date = None
        
        # Calculate costs
        base_costs = {
            'Unlimited_Premium': 90,
            'Unlimited_Standard': 70,
            'Limited_10GB': 55,
            'Limited_5GB': 45,
            'Limited_2GB': 35
        }
        
        base_cost = base_costs[plan_type]
        
        if family_plan_flag:
            if line_count == 2:
                cost_multiplier = 1.7
            elif line_count <= 4:
                cost_multiplier = line_count * 0.8
            else:
                cost_multiplier = line_count * 0.75
        else:
            cost_multiplier = 1.0
            
        monthly_cost = int(base_cost * cost_multiplier) + np.random.randint(-5, 10)
        
        # Data overage logic
        if 'Unlimited' in plan_type:
            data_overage_frequency = 0
        else:
            base_overage_prob = 0.2
            if customer['work_from_home_flag']:
                base_overage_prob += 0.15
            if customer['age'] < 30:
                base_overage_prob += 0.1
            elif customer['age'] > 60:
                base_overage_prob -= 0.05
            if customer['family_size'] >= 4:
                base_overage_prob += 0.1
                
            months_of_service = max(1, (current_date - start_date).days // 30)
            data_overage_frequency = np.random.binomial(months_of_service, base_overage_prob)
            
        # Device upgrade cycle
        if customer['income_bracket'] == 'High':
            device_upgrade_cycle = np.random.choice([12, 18, 24], p=[0.3, 0.4, 0.3])
        elif customer['income_bracket'] == 'Low':
            device_upgrade_cycle = np.random.choice([24, 36, 48], p=[0.2, 0.5, 0.3])
        else:
            device_upgrade_cycle = np.random.choice([18, 24, 36], p=[0.3, 0.5, 0.2])
            
        service = {
            'service_id': f"MOB_{str(service_counter).zfill(8)}",
            'customer_id': customer['customer_id'],
            'plan_type': plan_type,
            'line_count': line_count,
            'monthly_cost': monthly_cost,
            'data_overage_frequency': data_overage_frequency,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d') if end_date else None,
            'contract_type': contract_type,
            'contract_end_date': contract_end_date.strftime('%Y-%m-%d') if contract_end_date else None,
            'family_plan_flag': family_plan_flag,
            'device_upgrade_cycle': device_upgrade_cycle,
            'mobile_churn': 1 if end_date else 0
        }
        
        mobile_services.append(service)
        service_counter += 1
        
    df = pd.DataFrame(mobile_services)
    
    # Add calculated fields
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['contract_end_date'] = pd.to_datetime(df['contract_end_date'])
    
    df['mobile_tenure_days'] = np.where(
        df['end_date'].isnull(),
        (pd.to_datetime('today') - df['start_date']).dt.days,
        (df['end_date'] - df['start_date']).dt.days
    )
    
    print(f"Generated {len(df):,} mobile services")
    print(f"Active services: {len(df[df['end_date'].isna()]):,}")
    print(f"Contract distribution: {df['contract_type'].value_counts().to_dict()}")
    
    return df

# Keep the security function from the previous artifact (it was working fine)
def generate_security_devices_simple(customers_df):
    """
    Simplified security devices generation (original logic with minor fixes)
    """
    print("\nGenerating security devices...")
    
    security_devices = []
    device_counter = 1
    current_date = datetime.now()
    
    device_types = {
        'Smart_Doorbell': {'base_cost': 15, 'popularity': 0.4},
        'Outdoor_Camera': {'base_cost': 25, 'popularity': 0.3},
        'Indoor_Camera': {'base_cost': 20, 'popularity': 0.2},
        'Motion_Sensor': {'base_cost': 10, 'popularity': 0.15},
        'Smart_Lock': {'base_cost': 18, 'popularity': 0.12},
        'Window_Sensor': {'base_cost': 8, 'popularity': 0.10},
        'Security_Panel': {'base_cost': 35, 'popularity': 0.25}
    }
    
    for _, customer in customers_df.iterrows():
        # Security service adoption probability
        adoption_prob = 0.10
        
        # Crime rate significantly affects adoption
        if customer['neighborhood_crime_rate'] > 7.0:
            adoption_prob += 0.30
        elif customer['neighborhood_crime_rate'] > 5.0:
            adoption_prob += 0.15
        elif customer['neighborhood_crime_rate'] < 3.0:
            adoption_prob -= 0.05
            
        # Property value and type affect adoption
        if customer['property_value'] > 500000:
            adoption_prob += 0.20
        elif customer['property_value'] < 200000:
            adoption_prob -= 0.10
            
        if customer['home_type'] == 'Single_Family':
            adoption_prob += 0.15
        elif customer['home_type'] == 'Apartment':
            adoption_prob -= 0.20
            
        # Family characteristics
        if customer['life_stage'] in ['Young_Family', 'Established_Family']:
            adoption_prob += 0.15
            
        # Income affects adoption
        if customer['income_bracket'] == 'High':
            adoption_prob += 0.10
        elif customer['income_bracket'] == 'Low':
            adoption_prob -= 0.15
            
        # Age factor
        if 35 <= customer['age'] <= 55:
            adoption_prob += 0.05
        elif customer['age'] > 70:
            adoption_prob -= 0.05
            
        if np.random.random() > adoption_prob:
            continue
            
        # Generate installation date
        installation_date = current_date - timedelta(days=np.random.randint(30, 600))
        
        # Determine device count
        if customer['home_type'] == 'Single_Family' and customer['income_bracket'] in ['Upper_Middle', 'High']:
            device_count = np.random.choice([3, 4, 5, 6], p=[0.2, 0.3, 0.3, 0.2])
        elif customer['home_type'] in ['Townhouse', 'Single_Family']:
            device_count = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
        else:
            device_count = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
            
        # Select devices
        selected_devices = []
        
        # Smart doorbell first
        if np.random.random() < 0.8:
            selected_devices.append('Smart_Doorbell')
            
        # Add other devices
        remaining_devices = [d for d in device_types.keys() if d not in selected_devices]
        
        while len(selected_devices) < device_count and remaining_devices:
            device_probs = [device_types[d]['popularity'] for d in remaining_devices]
            total_prob = sum(device_probs)
            device_probs = [p/total_prob for p in device_probs]
            
            selected_device = np.random.choice(remaining_devices, p=device_probs)
            selected_devices.append(selected_device)
            remaining_devices.remove(selected_device)
            
        # Generate records for each device
        for device_type in selected_devices:
            base_cost = device_types[device_type]['base_cost']
            
            if customer['income_bracket'] == 'High':
                monthly_monitoring_cost = base_cost + np.random.randint(5, 15)
            else:
                monthly_monitoring_cost = base_cost + np.random.randint(-3, 8)
                
            # Some devices have no monthly cost
            if device_type in ['Motion_Sensor', 'Window_Sensor'] and np.random.random() < 0.3:
                monthly_monitoring_cost = 0
                
            # Status logic
            churn_prob = 0.25
            
            if monthly_monitoring_cost > 40:
                churn_prob += 0.10
                
            if customer['neighborhood_crime_rate'] > 6.0:
                churn_prob -= 0.08
                
            if customer['income_bracket'] == 'Low':
                churn_prob += 0.15
                
            if device_type == 'Smart_Doorbell':
                churn_prob -= 0.05
                
            device_days = (current_date - installation_date).days
            annual_churn_prob = churn_prob * (device_days / 365)
            
            if np.random.random() < annual_churn_prob:
                status = 'Inactive'
            else:
                status = 'Active'
                
            # App engagement
            base_app_opens = 20
            
            if device_type == 'Smart_Doorbell':
                app_multiplier = 2.0
            elif 'Camera' in device_type:
                app_multiplier = 1.5
            else:
                app_multiplier = 0.8
                
            if customer['age'] < 40:
                age_multiplier = 1.3
            elif customer['age'] > 60:
                age_multiplier = 0.7
            else:
                age_multiplier = 1.0
                
            if customer['work_from_home_flag']:
                wfh_multiplier = 1.4
            else:
                wfh_multiplier = 1.0
                
            app_opens_monthly = max(0, int(np.random.normal(
                base_app_opens * app_multiplier * age_multiplier * wfh_multiplier, 10)))
            
            # Alarm activations
            if 'Camera' in device_type or device_type == 'Motion_Sensor':
                base_alarms = customer['neighborhood_crime_rate'] * 0.5
                alarm_activations = max(0, np.random.poisson(base_alarms))
            else:
                alarm_activations = 0
                
            # False alarms
            if alarm_activations > 0:
                false_alarm_rate = np.random.uniform(0.6, 0.8)
                false_alarms = int(alarm_activations * false_alarm_rate)
            else:
                false_alarms = 0
                
            # Warranty status
            warranty_months = 24
            if device_days < warranty_months * 30:
                warranty_status = 'Active'
            else:
                warranty_status = 'Expired'
                
            device = {
                'device_id': f"SEC_{str(device_counter).zfill(8)}",
                'customer_id': customer['customer_id'],
                'device_type': device_type,
                'monthly_monitoring_cost': monthly_monitoring_cost,
                'installation_date': installation_date.strftime('%Y-%m-%d'),
                'status': status,
                'app_opens_monthly': app_opens_monthly,
                'alarm_activations': alarm_activations,
                'false_alarms': false_alarms,
                'warranty_status': warranty_status
            }
            
            security_devices.append(device)
            device_counter += 1
            
    df = pd.DataFrame(security_devices)
    print(f"Generated {len(df):,} security devices")
    print(f"Active devices: {len(df[df['status'] == 'Active']):,}")
    print(f"Unique customers with security: {df['customer_id'].nunique():,}")
    print(f"Device distribution: {df['device_type'].value_counts().to_dict()}")
    
    return df

# Main execution function
def generate_all_fixed_service_tables(customers_df):
    """Generate all service tables with fixed contract logic"""
    print("=== Generating Fixed Service Tables ===")
    
    # Generate each service table
    internet_df = generate_internet_services_fixed(customers_df)
    mobile_df = generate_mobile_services_fixed(customers_df)
    security_df = generate_security_devices_simple(customers_df)
    
    print(f"\n📊 Cross-Service Analysis:")
    
    # Customer service adoption
    customers_with_internet = set(internet_df['customer_id'].unique())
    customers_with_mobile = set(mobile_df['customer_id'].unique())
    customers_with_security = set(security_df['customer_id'].unique())
    
    print(f"Internet customers: {len(customers_with_internet):,}")
    print(f"Mobile customers: {len(customers_with_mobile):,}")
    print(f"Security customers: {len(customers_with_security):,}")
    
    # Bundle analysis
    triple_play = customers_with_internet & customers_with_mobile & customers_with_security
    dual_internet_mobile = customers_with_internet & customers_with_mobile - customers_with_security
    dual_internet_security = customers_with_internet & customers_with_security - customers_with_mobile
    dual_mobile_security = customers_with_mobile & customers_with_security - customers_with_internet
    
    print(f"\n📦 Bundle Analysis:")
    print(f"Triple play (all 3 services): {len(triple_play):,}")
    print(f"Internet + Mobile: {len(dual_internet_mobile):,}")
    print(f"Internet + Security: {len(dual_internet_security):,}")
    print(f"Mobile + Security: {len(dual_mobile_security):,}")
    
    # Contract validation
    print(f"\n📋 Contract Validation:")
    
    # Internet contract validation
    active_internet = internet_df[internet_df['end_date'].isna()]
    active_with_contracts = active_internet[active_internet['contract_end_date'].notna()]
    
    if len(active_with_contracts) > 0:
        past_contracts = active_with_contracts[active_with_contracts['contract_end_date'] < pd.to_datetime('today')]
        print(f"Internet: Active customers with past contract end dates: {len(past_contracts)}")
    
    # Mobile contract validation
    active_mobile = mobile_df[mobile_df['end_date'].isna()]
    active_mobile_contracts = active_mobile[active_mobile['contract_end_date'].notna()]
    
    if len(active_mobile_contracts) > 0:
        past_mobile_contracts = active_mobile_contracts[active_mobile_contracts['contract_end_date'] < pd.to_datetime('today')]
        print(f"Mobile: Active customers with past contract end dates: {len(past_mobile_contracts)}")
    
    # Revenue analysis
    total_internet_revenue = internet_df[internet_df['end_date'].isna()]['monthly_cost'].sum()
    total_mobile_revenue = mobile_df[mobile_df['end_date'].isna()]['monthly_cost'].sum()
    total_security_revenue = security_df[security_df['status'] == 'Active']['monthly_monitoring_cost'].sum()
    
    print(f"\n💰 Monthly Revenue Analysis (Active Services Only):")
    print(f"Internet MRR: ${total_internet_revenue:,.0f}")
    print(f"Mobile MRR: ${total_mobile_revenue:,.0f}")
    print(f"Security MRR: ${total_security_revenue:,.0f}")
    print(f"Total MRR: ${total_internet_revenue + total_mobile_revenue + total_security_revenue:,.0f}")
    
    print(f"\n✅ Fixed service table generation completed successfully!")
    
    return internet_df, mobile_df, security_df

# Usage example:
# internet_df, mobile_df, security_df = generate_all_fixed_service_tables(customers_df)