import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import google.generativeai as genai
import json
import time
import os
from typing import Dict, List, Tuple, Optional
import warnings
import re
warnings.filterwarnings("ignore")

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
transcript_model = genai.GenerativeModel('gemini-2.5-flash')
evaluation_model = genai.GenerativeModel('gemini-2.5-flash-lite')

class EnhancedCallCenterDataGenerator:
    """Two-stage call center data generator: natural transcripts + focused evaluation"""
    
    def __init__(self, customers_df, internet_df, mobile_df, security_df):
        self.customers_df = customers_df
        self.internet_df = internet_df
        self.mobile_df = mobile_df
        self.security_df = security_df
        
        # Agent pool with skill levels (for context only, not predetermined outcomes)
        self.agents = {
            'agent_001': {'name': 'Sarah Mitchell', 'specialization': 'general', 'skill_level': 0.8},
            'agent_002': {'name': 'David Chen', 'specialization': 'technical', 'skill_level': 0.9},
            'agent_003': {'name': 'Maria Rodriguez', 'specialization': 'billing', 'skill_level': 0.8},
            'agent_004': {'name': 'James Thompson', 'specialization': 'retention', 'skill_level': 0.7},
            'agent_005': {'name': 'Lisa Wang', 'specialization': 'sales', 'skill_level': 0.6},
            'agent_006': {'name': 'Michael Brown', 'specialization': 'technical', 'skill_level': 0.9},
            'agent_007': {'name': 'Jennifer Davis', 'specialization': 'billing', 'skill_level': 0.8},
            'agent_008': {'name': 'Robert Kim', 'specialization': 'general', 'skill_level': 0.7},
            'agent_009': {'name': 'Ashley Johnson', 'specialization': 'retention', 'skill_level': 0.8},
            'agent_010': {'name': 'Carlos Martinez', 'specialization': 'escalation', 'skill_level': 0.9}
        }
        
        # Scenario weights - configurable
        self.default_weights = {
            'billing_inquiry': 0.16,
            'technical_support': 0.18,
            'service_upgrade': 0.07,
            'cross_sell_mobile': 0.10,
            'cross_sell_security': 0.08,
            'cross_sell_internet': 0.05,
            'upsell_security_devices': 0.06,
            'service_cancellation': 0.06,
            'contract_renewal': 0.05,
            'payment_assistance': 0.04,
            'complaint_resolution': 0.08,
            'service_downgrade': 0.03,
            'general_inquiry': 0.02,
            'equipment_replacement': 0.02
        }
        
        # Scenario descriptions for context
        self.scenarios = {
            'billing_inquiry': 'Customer questioning bill charges, payment issues, or billing discrepancies',
            'technical_support': 'Technical issues affecting service functionality',
            'service_upgrade': 'Customer wanting to upgrade their current service plan',
            'cross_sell_mobile': 'Existing customer interested in adding mobile services',
            'cross_sell_security': 'Existing customer interested in home security services',
            'cross_sell_internet': 'Mobile/security customer interested in internet services',
            'upsell_security_devices': 'Existing security customer wanting additional devices',
            'service_cancellation': 'Customer wanting to cancel service',
            'contract_renewal': 'Contract expiration and renewal discussions',
            'payment_assistance': 'Customer needing help with payment arrangements',
            'complaint_resolution': 'Formal complaints about service or experience',
            'service_downgrade': 'Customer wanting to reduce service',
            'general_inquiry': 'General questions about services or account',
            'equipment_replacement': 'Customer needs equipment replaced or upgraded'
        }
    
    def get_customer_services(self, customer_id: str) -> Dict:
        """Get comprehensive customer service information"""
        context = {'customer_info': {}, 'services': {}, 'monthly_total': 0, 'issue_count': 0}
        
        # Customer demographics
        customer_row = self.customers_df[self.customers_df['customer_id'] == customer_id]
        if not customer_row.empty:
            context['customer_info'] = customer_row.iloc[0].to_dict()
        
        # Internet service
        internet_row = self.internet_df[self.internet_df['customer_id'] == customer_id]
        if not internet_row.empty:
            service = internet_row.iloc[0].to_dict()
            context['services']['internet'] = service
            context['monthly_total'] += service['monthly_cost']
            context['issue_count'] += service.get('speed_complaints', 0) + service.get('outage_count', 0)
        
        # Mobile service
        mobile_row = self.mobile_df[self.mobile_df['customer_id'] == customer_id]
        if not mobile_row.empty:
            service = mobile_row.iloc[0].to_dict()
            context['services']['mobile'] = service
            context['monthly_total'] += service['monthly_cost']
            context['issue_count'] += service.get('data_overage_frequency', 0)
        
        # Security service
        security_rows = self.security_df[self.security_df['customer_id'] == customer_id]
        if not security_rows.empty:
            devices = security_rows.to_dict('records')
            context['services']['security'] = devices
            context['monthly_total'] += sum(d.get('monthly_monitoring_cost', 0) for d in devices)
        
        return context
    
    def select_call_scenario(self, customer_context: Dict, scenario_weights: Dict) -> str:
        """Select primary scenario based on customer situation"""
        available_services = list(customer_context['services'].keys())
        
        # Adjust weights based on customer situation
        adjusted_weights = {}
        for scenario, weight in scenario_weights.items():
            if scenario not in self.scenarios:
                continue
                
            # Boost based on customer history
            if scenario == 'technical_support' and customer_context['issue_count'] > 2:
                weight *= 2.0
            elif scenario == 'complaint_resolution' and customer_context['issue_count'] > 3:
                weight *= 1.5
            elif 'cross_sell' in scenario:
                # Only allow if customer doesn't have that service
                if 'mobile' in scenario and 'mobile' in available_services:
                    weight = 0
                elif 'security' in scenario and 'security' in available_services:
                    weight = 0
                elif 'internet' in scenario and 'internet' in available_services:
                    weight = 0
            
            adjusted_weights[scenario] = weight
        
        # Select primary scenario
        if sum(adjusted_weights.values()) == 0:
            return 'general_inquiry'
        else:
            scenarios = list(adjusted_weights.keys())
            weights = list(adjusted_weights.values())
            total = sum(weights)
            probs = [w/total for w in weights]
            return np.random.choice(scenarios, p=probs)
    
    def select_agent(self, scenario: str) -> str:
        """Select appropriate agent based on scenario and specialization"""
        if scenario == 'technical_support':
            preferred_agents = [aid for aid, info in self.agents.items() if info['specialization'] == 'technical']
        elif scenario == 'billing_inquiry':
            preferred_agents = [aid for aid, info in self.agents.items() if info['specialization'] == 'billing']
        elif 'cross_sell' in scenario or scenario == 'service_cancellation':
            preferred_agents = [aid for aid, info in self.agents.items() if info['specialization'] in ['sales', 'retention']]
        elif scenario == 'complaint_resolution':
            preferred_agents = [aid for aid, info in self.agents.items() if info['specialization'] == 'escalation']
        else:
            preferred_agents = list(self.agents.keys())
        
        return random.choice(preferred_agents)
    
    def build_services_context(self, customer_context: Dict) -> str:
        """Build readable services description for prompts"""
        services_text = ""
        for service_type, service_data in customer_context['services'].items():
            if service_type == 'internet':
                services_text += f"Internet: {service_data['plan_tier']} ({service_data['speed_mbps']} Mbps) - ${service_data['monthly_cost']}/month. "
                if service_data.get('speed_complaints', 0) > 0:
                    services_text += f"Recent speed complaints: {service_data['speed_complaints']}. "
                if service_data.get('outage_count', 0) > 0:
                    services_text += f"Recent outages: {service_data['outage_count']}. "
            elif service_type == 'mobile':
                services_text += f"Mobile: {service_data['plan_type']} - {service_data['line_count']} lines - ${service_data['monthly_cost']}/month. "
                if service_data.get('data_overage_frequency', 0) > 0:
                    services_text += f"Recent data overages: {service_data['data_overage_frequency']}. "
            elif service_type == 'security':
                device_count = len(service_data)
                services_text += f"Security: {device_count} devices installed with monitoring. "
        return services_text.strip()
    
    def generate_natural_transcript(self, customer_context: Dict, primary_scenario: str, 
                                  agent_id: str, call_date: str) -> str:
        """Stage 1: Generate natural conversation transcript using Gemini 2.5 Flash"""
        
        agent_info = self.agents[agent_id]
        agent_name = agent_info['name']
        services_text = self.build_services_context(customer_context)
        
        # Create comprehensive prompt for natural conversation
        prompt = f"""
Generate a realistic customer service call transcript for TriLink Telecom. Make this authentic - not all calls go perfectly.

CUSTOMER PROFILE:
- Customer ID: {customer_context['customer_info'].get('customer_id', 'Unknown')}
- Age: {customer_context['customer_info'].get('age', 'Unknown')}
- Income Level: {customer_context['customer_info'].get('income_bracket', 'Unknown')}
- Current Services: {services_text}
- Monthly Bill Total: ${customer_context['monthly_total']:.0f}
- Service Issues History: {customer_context['issue_count']} recent complaints/problems

CALL SETUP:
- Date: {call_date}
- Agent: {agent_name} ({agent_id})
- Agent Specialization: {agent_info['specialization']} specialist
- Agent Experience Level: {agent_info['skill_level']}/1.0 (affects their knowledge and approach)
- Call Reason: {self.scenarios[primary_scenario]}

INSTRUCTIONS:
1. Create a natural, flowing conversation between customer and agent
2. Customer should explain their specific issue using realistic details from their account
3. Agent should respond based on their specialization and experience level
4. Include realistic dialogue with natural pauses, clarifications, and human elements
5. Agent should reference actual customer service information when investigating
6. Let the conversation develop naturally - don't force any specific outcome
7. Include technical details, specific pricing, and service plan information where relevant
8. Make it feel like a real phone conversation with natural back-and-forth
9. Target length: 500-800 words for a complete call
10. End the call naturally when the issue is addressed or needs escalation
11. Create an authentic conversation - problems don't always get solved perfectly
12. Agent may not have immediate solutions or authority to resolve everything. For example in cases like this:
        - Technical issues that can't be immediately resolved
        - Billing disputes that require investigation
        - Policy limitations that prevent desired solutions
        - System outages affecting service ability
        - Customer frustration with repeated problems
        - Agent limitations in authority or knowledge
        - Need for escalation or follow-up calls
        - Partial resolutions that don't fully satisfy customer
13. Customer may remain frustrated if issues persist or solutions are inadequate


The conversation should feel authentic and unscripted. Focus on realistic problem-solving dialogue.


Generate the complete call transcript:"""

        try:
            response = transcript_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating transcript: {e}")
            return self._generate_fallback_transcript(customer_context, primary_scenario, agent_name)
    
    def evaluate_call_performance(self, transcript: str) -> Tuple[int, bool]:
        """Stage 2: Evaluate call using Gemini 2.5 Flash Lite"""
        
        evaluation_prompt = f"""Analyze this customer service call transcript and provide a REALISTIC evaluation. Be critical and honest - not all calls go well in real call centers.

CALL TRANSCRIPT:
{transcript}

EVALUATION INSTRUCTIONS:
Rate this call on a scale of 0-10 considering both agent performance AND customer satisfaction. BE REALISTIC - most calls are not perfect.

RATING SCALE (BE HONEST):
- 0-2: Poor (customer frustrated/angry, major issues unresolved, poor agent performance)
- 3-4: Below Average (customer somewhat unsatisfied, partial resolution, agent struggled)
- 5-6: Average (customer neutral/okay, basic resolution, standard agent performance)
- 7-8: Good (customer satisfied, effective resolution, agent performed well)
- 9-10: Excellent (customer delighted, exceptional service, outstanding agent work) - RARE

IMPORTANT: In real call centers, only 20-30% of calls get 9-10 ratings. Most calls are 5-7. Don't be overly generous with ratings.

Look for problems like:
- Customer still frustrated at end
- Issue not fully resolved
- Agent made mistakes or was unprofessional
- Customer had to call back later
- Long hold times or transfers mentioned
- Agent couldn't provide what customer wanted

Also determine if the call was successful (true/false):
- True: Customer's main issue was resolved OR customer accepted a reasonable solution/offer
- False: Customer's issue remains unresolved OR customer was unsatisfied with outcome

Respond with ONLY this JSON format:
{{
    "overall_rating": <integer 0-10>,
    "call_successful": <true or false>
}}"""

        try:
            response = evaluation_model.generate_content(evaluation_prompt)
            return self._parse_evaluation_response(response.text)
        except Exception as e:
            print(f"Error evaluating call: {e}")
            # Fallback to neutral evaluation
            return 5, True
    
    def _parse_evaluation_response(self, response_text: str) -> Tuple[int, bool]:
        """Parse evaluation response and extract rating and success"""
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{[^}]*\}', response_text)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                rating = int(data.get('overall_rating', 5))
                success = bool(data.get('call_successful', True))
                return max(0, min(10, rating)), success
            else:
                # Fallback parsing
                lines = response_text.strip().split('\n')
                rating = 5
                success = True
                for line in lines:
                    if 'rating' in line.lower() and any(char.isdigit() for char in line):
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            rating = int(numbers[0])
                    if 'success' in line.lower():
                        success = 'true' in line.lower() or 'yes' in line.lower()
                return max(0, min(10, rating)), success
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            return 5, True
    
    def _generate_fallback_transcript(self, customer_context: Dict, scenario: str, agent_name: str) -> str:
        """Fallback transcript if API fails"""
        return f"""Agent: Hello, thank you for calling TriLink customer service. This is {agent_name}, how can I help you today?

Customer: Hi, I'm calling about my {scenario.replace('_', ' ')} issue with my account.

Agent: I'd be happy to help you with that. Let me look into your account details.

Customer: Thank you, I hope we can get this resolved quickly.

Agent: I've reviewed your account and I can see the issue you're referring to. Let me work on getting this resolved for you.

Customer: That would be great, I appreciate your help.

Agent: I've been able to address your concern. Is there anything else I can help you with today?

Customer: No, that covers everything. Thank you for your assistance.

Agent: You're welcome! Thank you for calling TriLink. Have a great day!"""
    
    def generate_call_dataset(self, start_date: str = "2025-01-01", end_date: str = "2025-09-01", 
                            num_calls: int = 1000, scenario_weights: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate complete call center dataset using two-stage approach
        
        Args:
            start_date: Start date for calls (YYYY-MM-DD)
            end_date: End date for calls (YYYY-MM-DD) 
            num_calls: Number of calls to generate
            scenario_weights: Optional custom scenario weights dict
        
        Returns:
            DataFrame with call center data
        """
        
        if scenario_weights is None:
            scenario_weights = self.default_weights
        
        print(f"\n🎯 ENHANCED CALL GENERATION STARTING")
        print(f"📊 Target: {num_calls} calls from {start_date} to {end_date}")
        print(f"🤖 Using Two-Stage Approach:")
        print(f"   Stage 1: Natural transcripts (Gemini 2.5 Flash)")
        print(f"   Stage 2: Performance evaluation (Gemini 2.5 Flash Lite)")
        print("=" * 60)
        
        # Convert dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = (end_dt - start_dt).days
        
        # FAST customer filtering - no loops!
        print(f"🔍 Fast analysis of {len(self.customers_df)} customers...")

        internet_customers = set(self.internet_df['customer_id'].unique()) if not self.internet_df.empty else set()
        mobile_customers = set(self.mobile_df['customer_id'].unique()) if not self.mobile_df.empty else set()
        security_customers = set(self.security_df['customer_id'].unique()) if not self.security_df.empty else set()

        customers_with_services = list(internet_customers | mobile_customers | security_customers)

        print(f"✅ Found {len(customers_with_services)} customers with services")
        print(f"   📡 Internet: {len(internet_customers)} customers")
        print(f"   📱 Mobile: {len(mobile_customers)} customers") 
        print(f"   🏠 Security: {len(security_customers)} customers")

        # Pre-build contexts for a sample
        sample_size = min(num_calls * 2, len(customers_with_services))
        sample_customers = random.sample(customers_with_services, sample_size)

        customer_contexts = {}
        for idx, customer_id in enumerate(sample_customers):
            if idx % 100 == 0:
                print(f"   Building context {idx}/{sample_size}...")
            customer_contexts[customer_id] = self.get_customer_services(customer_id)
        
        print(f"🔍 Found {len(customers_with_services)} customers with services")
        
        calls = []
        successful_calls = 0
        api_errors = 0
        start_time = time.time()
        
        for i in range(num_calls):
            if i % 25 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (num_calls - i) * avg_time
                print(f"⚡ Progress: {i}/{num_calls} ({i/num_calls*100:.1f}%) - ETA: {remaining/60:.1f} minutes")
            
            try:
                print(f"\n🔄 Starting call {i+1}/{num_calls}...")
                
                # Select customer and get context (from pre-built contexts)
                customer_id = random.choice(list(customer_contexts.keys()))
                print(f"   👤 Selected customer: {customer_id}")
                customer_context = customer_contexts[customer_id]
                
                # Generate call date and time
                days_offset = random.randint(0, date_range)
                call_date = start_dt + timedelta(days=days_offset)
                call_time = f"{random.randint(8, 17):02d}:{random.randint(0, 59):02d}"
                
                # Select scenario and agent
                primary_scenario = self.select_call_scenario(customer_context, scenario_weights)
                agent_id = self.select_agent(primary_scenario)
                print(f"   📋 Scenario: {primary_scenario} | Agent: {agent_id}")
                
                # Stage 1: Generate natural transcript
                print(f"   🎤 Generating transcript...", end="", flush=True)
                transcript = self.generate_natural_transcript(
                    customer_context, primary_scenario, agent_id, call_date.strftime('%Y-%m-%d')
                )
                print(" ✅", flush=True)
                
                # Brief delay between API calls
                time.sleep(0.2)
                
                # Stage 2: Evaluate performance
                print(f"   📊 Evaluating call...", end="", flush=True)
                overall_rating, call_successful = self.evaluate_call_performance(transcript)
                print(f" ✅ (Rating: {overall_rating}/10, Success: {call_successful})", flush=True)
                
                # Create call record
                call_record = {
                    'call_id': f"CALL_{str(i + 1).zfill(6)}",
                    'customer_id': customer_id,
                    'call_date': call_date.strftime('%Y-%m-%d'),
                    'call_time': call_time,
                    'agent_id': agent_id,
                    'agent_name': self.agents[agent_id]['name'],
                    'primary_scenario': primary_scenario,
                    'call_transcript': transcript,
                    'overall_rating': overall_rating,
                    'call_successful': call_successful,
                    'customer_monthly_spend': customer_context['monthly_total'],
                    'customer_service_count': len(customer_context['services']),
                    'customer_issue_history': customer_context['issue_count']
                }
                
                calls.append(call_record)
                successful_calls += 1
                
                # Brief delay for API rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                api_errors += 1
                print(f"❌ Error generating call {i+1}: {e}")
                if api_errors > 20:
                    print("⚠️  Too many errors - stopping generation")
                    break
                continue
        
        # Final completion status
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("🎉 ENHANCED CALL GENERATION COMPLETE!")
        print("="*60)
        print(f"📊 Final Results:")
        print(f"   ✅ Successfully generated: {successful_calls}/{num_calls} calls")
        print(f"   ❌ Failed generations: {api_errors}")
        print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"   ⚡ Average time per call: {total_time/max(successful_calls,1):.1f} seconds")
        
        if successful_calls == 0:
            print("❌ No calls were generated successfully. Check your API configuration.")
            return pd.DataFrame()
        
        # Create DataFrame
        calls_df = pd.DataFrame(calls)
        
        # Display enhanced summary
        print(f"\n📞 CALL CENTER DATASET SUMMARY")
        print(f"Total Calls: {len(calls_df)}")
        print(f"Date Range: {calls_df['call_date'].min()} to {calls_df['call_date'].max()}")
        print(f"Success Rate: {calls_df['call_successful'].mean():.1%}")
        print(f"Average Rating: {calls_df['overall_rating'].mean():.2f}/10.0")
        print(f"Rating Distribution:")
        for rating in sorted(calls_df['overall_rating'].unique()):
            count = (calls_df['overall_rating'] == rating).sum()
            print(f"  {rating}/10: {count} calls ({count/len(calls_df)*100:.1f}%)")
        
        print(f"\nTop Scenarios:")
        scenario_counts = calls_df['primary_scenario'].value_counts().head(5)
        for scenario, count in scenario_counts.items():
            avg_rating = calls_df[calls_df['primary_scenario'] == scenario]['overall_rating'].mean()
            print(f"  {scenario.replace('_', ' ').title()}: {count} calls (avg rating: {avg_rating:.1f})")
        
        return calls_df

# Main function to use
def create_enhanced_trilink_calls(customers_df, internet_df, mobile_df, security_df,
                                start_date="2025-07-01", end_date="2025-09-01", 
                                num_calls=1000, scenario_weights=None):
    """
    Create TriLink call center dataset using enhanced two-stage generation
    
    Args:
        customers_df, internet_df, mobile_df, security_df: Your service dataframes
        start_date: Start date for calls (YYYY-MM-DD)
        end_date: End date for calls (YYYY-MM-DD)
        num_calls: Number of calls to generate
        scenario_weights: Optional dict to customize scenario distribution
        
    Returns:
        DataFrame with call data optimized for embeddings and ML applications
    """
    generator = EnhancedCallCenterDataGenerator(customers_df, internet_df, mobile_df, security_df)
    calls_df = generator.generate_call_dataset(start_date, end_date, num_calls, scenario_weights)
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"enhanced_trilink_calls_{start_date}_{end_date}_{timestamp}.csv"
    calls_df.to_csv(filename, index=False)
    print(f"\n💾 Saved enhanced dataset: {filename}")
    
    return calls_df

# Example usage:
# calls_df = create_enhanced_trilink_calls(
#     customers_df, internet_df, mobile_df, security_df,
#     start_date="2025-07-01", 
#     end_date="2025-09-01",
#     num_calls=1000
# )