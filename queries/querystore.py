import os
from dotenv import load_dotenv
load_dotenv()
project_id=os.environ.get("project_id")
dataset_id=os.environ.get("database_name")



class QueryStore:
    """
    Module for generating BigQuery AI-powered retention scripts for mobile customers.
    Combines churn prediction, vector search, and personalized script generation.
    """
    
    def __init__(self):
        self.project_id = "trilink-472019"
        self.dataset_id = "database"
        
    def get_retention_call_script_mobile_query(self, customer_id):
        """
        Generate the complete BigQuery retention script query for a specific customer.
        
        Args:
            customer_id (str): The customer ID to analyze
            
        Returns:
            str: Complete BigQuery SQL with AI.GENERATE for retention script
        """
        query = f"""
-- Base customer churn predictions for multiple customers
WITH customer_churn_base AS (
  SELECT
    customer_id,
    predicted_mobile_churn,
    ROUND(probability * 100, 1) AS churn_probability_percent,
    age,
    plan_type,
    monthly_cost,
    contract_type,
    tenure_years,
    data_overage_frequency,
    family_plan_flag,
    household_income,
    family_size,
    home_ownership,
    work_from_home_flag,
    education_level,
    life_stage,
    income_per_person,
    cost_per_line,
    fiber_availability,
    neighborhood_crime_rate,
    
    -- Risk category
    CASE
      WHEN probability >= 0.8 THEN 'CRITICAL RISK'
      WHEN probability >= 0.6 THEN 'HIGH RISK'
      WHEN probability >= 0.4 THEN 'MEDIUM RISK'
      ELSE 'LOW RISK'
    END AS risk_category,
    
    top_feature_attributions
  FROM (
    SELECT *
    FROM ML.EXPLAIN_PREDICT(
      MODEL `{self.project_id}.{self.dataset_id}.mobile_churn_predictor`,
      (SELECT * FROM `{self.project_id}.{self.dataset_id}.mobile_churn_data`
       WHERE customer_id='{customer_id}'
      )
    )
  )
),

-- Security services aggregation
customer_security AS (
  SELECT 
    customer_id,
    STRING_AGG(device_type, ', ' ORDER BY device_type) AS device_types_concat,
    COUNT(device_type) AS total_security_devices
  FROM `{self.project_id}.{self.dataset_id}.security_df`
  WHERE customer_id IN (SELECT customer_id FROM customer_churn_base)
  GROUP BY customer_id
),

-- Active internet services
customer_internet AS (
  SELECT 
    customer_id,
    speed_mbps AS internet_speed,
    plan_tier AS internet_plan,
    contract_type AS internet_contract_type,
    internet_tenure_days
  FROM `{self.project_id}.{self.dataset_id}.internet_df`
  WHERE customer_id IN (SELECT customer_id FROM customer_churn_base)
    AND internet_churn = 0
),

-- Active mobile services (non-churned)
customer_mobile AS (
  SELECT 
    customer_id,
    plan_type AS current_mobile_plan,
    contract_type AS current_mobile_contract_type,
    data_overage_frequency AS current_mobile_data_overage,
    monthly_cost AS current_mobile_cost
  FROM `{self.project_id}.{self.dataset_id}.mobile_df`
  WHERE customer_id IN (SELECT customer_id FROM customer_churn_base)
    AND mobile_churn = 0
),

-- Problem descriptions for business solution matching (AI-generated)
customer_problems AS (
  SELECT 
    cb.customer_id,
    AI.GENERATE(
      CONCAT(
        'Analyze this mobile customer situation and identify their key problems/pain points in under 150 words. Start with "Customer is facing...":\\n\\n',
        'Customer Profile:\\n',
        '- Age: ', CAST(cb.age AS STRING), ' years (', cb.life_stage, ')\\n',
        '- Income: $', CAST(cb.household_income AS STRING), ' household, $', CAST(CAST(cb.income_per_person AS INT64) AS STRING), ' per person\\n',
        '- Family: ', CAST(cb.family_size AS STRING), ' people, ', cb.home_ownership, '\\n',
        '- Education: ', cb.education_level, '\\n',
        '- Works from home: ', CAST(cb.work_from_home_flag AS STRING), '\\n',
        'Mobile Service:\\n',
        '- Plan: ', cb.plan_type, '\\n',
        '- Cost: $', CAST(cb.monthly_cost AS STRING), '/month ($', CAST(CAST(cb.cost_per_line AS INT64) AS STRING), ' per line)\\n',
        '- Contract: ', cb.contract_type, '\\n',
        '- Tenure: ', CAST(cb.tenure_years AS STRING), ' years\\n',
        '- Data overages: ', CAST(cb.data_overage_frequency AS STRING), ' per month\\n',
        '- Family plan: ', CAST(cb.family_plan_flag AS STRING), '\\n',
        '- Fiber available: ', CAST(cb.fiber_availability AS STRING), '\\n\\n',
        'Churn Analysis:\\n',
        '- Risk: ', CAST(cb.churn_probability_percent AS STRING), '% (', cb.risk_category, ')\\n',
        '- Key factors: ', ARRAY_TO_STRING(
          ARRAY(
            SELECT CONCAT(feature, ': ', CAST(ROUND(attribution, 3) AS STRING))
            FROM UNNEST(cb.top_feature_attributions)
            ORDER BY ABS(attribution) DESC
            LIMIT 5
          ), ', '
        ), '\\n\\n',
        'Task: Write a detailed problem analysis identifying:\\n',
        'Why they are at risk of churning? What could be the possible reasons and or frustration customer is facing\\n',
        'Format as a comprehensive problem statement for vector similarity matching with business solutions later.'
      ),
      connection_id => 'us.vertex-ai-connection',
      endpoint => 'gemini-2.5-flash'
    ).result AS problem_description
  FROM customer_churn_base cb
),

-- Generate embeddings for customer problems
customer_problem_embeddings AS (
  SELECT *
  FROM ML.GENERATE_EMBEDDING(
    MODEL `{self.project_id}.{self.dataset_id}.text_embedding_005_model`,
    (SELECT 
       problem_description AS content,
       customer_id
     FROM customer_problems),
    STRUCT(TRUE AS flatten_json_output, 'RETRIEVAL_DOCUMENT' AS task_type)
  )
),

-- Business solution matching
business_solutions AS (
  SELECT 
    vs.query.customer_id,
    vs.base.problem AS similar_matched_problem,
    vs.base.solution AS recommended_solution,
    vs.base.sector AS sector,
    vs.distance AS solution_match_score
  FROM VECTOR_SEARCH(
    TABLE `{self.project_id}.{self.dataset_id}.business_solution_embeddings`,
    'problem_embedding',
    TABLE customer_problem_embeddings,
    'ml_generate_embedding_result',
    top_k => 1,
    distance_type => 'COSINE'
  ) AS vs
),

-- Consolidated customer data
customer_consolidated AS (
  SELECT 
    cb.*,
    
    -- Security services
    COALESCE(cs.device_types_concat, 'None') AS device_types_concat,
    COALESCE(cs.total_security_devices, 0) AS total_security_devices,
    CASE WHEN cs.customer_id IS NOT NULL THEN 1 ELSE 0 END AS security_customer,
    
    -- Internet services
    ci.internet_speed,
    ci.internet_plan,
    ci.internet_contract_type,
    ci.internet_tenure_days,
    CASE 
      WHEN ci.customer_id IS NULL THEN 'No internet service'
      ELSE 'Active internet customer' 
    END AS internet_customer_status,
    
    -- Mobile services (from churn table vs active table)
    cm.current_mobile_plan,
    cm.current_mobile_contract_type,
    cm.current_mobile_data_overage,
    cm.current_mobile_cost,
    CASE 
      WHEN cm.customer_id IS NULL THEN 'Mobile service at risk'
      ELSE 'Active mobile customer'
    END AS mobile_customer_status,
    
    -- Business solutions
    bs.similar_matched_problem,
    bs.recommended_solution,
    bs.sector,
    bs.solution_match_score
    
  FROM customer_churn_base cb
  LEFT JOIN customer_security cs ON cb.customer_id = cs.customer_id
  LEFT JOIN customer_internet ci ON cb.customer_id = ci.customer_id
  LEFT JOIN customer_mobile cm ON cb.customer_id = cm.customer_id
  LEFT JOIN business_solutions bs ON cb.customer_id = bs.customer_id
)

-- Final output with retention script generation
SELECT
  churn_probability_percent,
  risk_category,
  plan_type,
  monthly_cost,
  internet_customer_status,
  mobile_customer_status,
  security_customer,
  
  -- Use the AI-generated problem description from solution matching
  cp.problem_description AS problem_analysis,
  similar_matched_problem,
  recommended_solution,
  solution_match_score,

  AI.GENERATE(
    CONCAT(
      'Create a LIVE CALL retention script for a call center agent handling this at-risk customer RIGHT NOW. Focus on PROBLEM-SOLVING FIRST, then strategic value-building:\\n\\n',
      
      '=== CUSTOMER OVERVIEW ===\\n',
      'Churn Risk: ', CAST(churn_probability_percent AS STRING), '% (', risk_category, ')\\n',
      'Profile: ', CAST(age AS STRING), '-year-old ', life_stage, '\\n',
      'Family: ', CAST(family_size AS STRING), ' people, ', home_ownership, '\\n',
      'Income: $', CAST(household_income AS STRING), ' household\\n',
      'Work Setup: ', CASE WHEN work_from_home_flag THEN 'Works from home' ELSE 'Traditional workplace' END, '\\n\\n',
      
      '=== CURRENT SERVICE PORTFOLIO ===\\n',
      'Mobile Service: ', CAST(tenure_years AS STRING), ' years with us\\n',
      '- Plan: ', plan_type, ' ($', CAST(monthly_cost AS STRING), '/month)\\n',
      '- Contract: ', contract_type, '\\n',
      '- Data Issues: ', CAST(data_overage_frequency AS STRING), ' overages/month\\n',
      '- Family Plan: ', CAST(family_plan_flag AS STRING), '\\n\\n',
      
      'Internet Service: ', internet_customer_status, '\\n',
      CASE WHEN internet_customer_status = 'Active internet customer' THEN
        CONCAT('- Plan: ', internet_plan, ' (', CAST(internet_speed AS STRING), ' Mbps)\\n',
               '- Contract: ', internet_contract_type, '\\n')
        ELSE CONCAT('- Opportunity: Fiber available = ', CAST(fiber_availability AS STRING), '\\n') END,
      
      'Security Service: ', CASE WHEN security_customer = 1 
        THEN CONCAT('Active (', CAST(total_security_devices AS STRING), ' devices: ', device_types_concat, ')')
        ELSE 'Not subscribed' END, '\\n',
        CASE WHEN security_customer = 0 THEN 
        CONCAT('- Area Crime Rate: ', neighborhood_crime_rate, '\\n') ELSE '' END, '\\n',

      
      '=== CHURN RISK FACTORS ===\\n',
      'Risk Level: ', risk_category, ' (', CAST(churn_probability_percent AS STRING), '%)\\n',
      'Key Risk Drivers:\\n',
      ARRAY_TO_STRING(
        ARRAY(
          SELECT CONCAT('• ', feature, ': ', 
            CASE WHEN attribution > 0 THEN 'Increasing churn risk' 
                 ELSE 'Reducing churn risk' END,
            ' (', CAST(ROUND(attribution, 3) AS STRING), ')')
          FROM UNNEST(top_feature_attributions)
          ORDER BY ABS(attribution) DESC
          LIMIT 5
        ), '\\n'
      ), '\\n\\n',
      
      '=== BUSINESS SOLUTION GUIDANCE ===\\n',
      'From the business solution book, we found a problem most similar to this customers issue: ', COALESCE(similar_matched_problem, 'Standard mobile service retention'), '\\n',
      'And the business solution to this was: ', COALESCE(recommended_solution, 'Personalized value demonstration and service optimization'), '\\n',
      
      '=== AVAILABLE SOLUTIONS & UPGRADES ===\\n',
      'MOBILE PLANS (solve data overage issues):\\n',
      '• Unlimited Standard: $70/month (save on overages!)\\n',
      '• Unlimited Premium: $90/month (includes Disney+, hotspot)\\n\\n',
      
      'INTERNET SOLUTIONS (solve WFH/speed issues):\\n',
      '• Standard: 100 Mbps at $70/month\\n',
      '• Premium Gig: 1000 Mbps at $100/month (perfect for WFH)\\n\\n',
      
      'SECURITY OPTIONS (solve safety concerns):\\n',
      '• Smart Security: Doorbell + Camera ($200 + free install)\\n',
      '• Complete Protection: Full monitoring ($30/month + $50 off equipment)\\n',
      '• Starter Kit: Motion + Window sensors ($60)\\n\\n',
      
      '=== RETENTION SCRIPT REQUIREMENTS ===\\n',
      '**CALL STRUCTURE (Follow this order):**\\n',
      '1. ACKNOWLEDGE & LISTEN (30 seconds)\\n',
      '   - Thank them for their ', CAST(tenure_years AS STRING), '-year loyalty\\n',
      '   - Ask: "What brings you to call us today?" (even if you know)\\n',
      '   - Listen actively, show empathy\\n\\n',
      
      '2. PROBLEM-SOLVING FIRST (2-3 minutes)\\n',
      '   - Address their immediate pain points\\n',
      '   - Use the proven solution approach provided\\n',
      '   - Offer specific fixes before any sales pitches\\n',
      '   - Show how we can solve their actual problems\\n\\n',
      
      '3. VALUE REINFORCEMENT (1 minute)\\n',
      '   - Highlight their customer history and status\\n',
      '   - Mention loyalty benefits they have\\n',
      '   - Reference bundling savings they could get\\n\\n',
      
      '4. STRATEGIC CROSS-SELL (if appropriate, 1-2 minutes)\\n',
      '   - ONLY suggest services that solve their problems\\n',
      '   - Focus on services they DONT have but NEED\\n',
      '   - Frame as problem-solving, not selling\\n\\n',
      
      '5. RETENTION OFFER & CLOSE (1 minute)\\n',
      '   - Provide meaningful retention incentive\\n',
      '   - Set clear next steps\\n',
      '   - Get commitment\\n\\n',
      
      '**RETENTION OFFER GUIDELINES:**\\n',
      '• Maximum $20 discount on existing services\\n',
      '• FREE first month on NEW services (internet/mobile upgrades)\\n',
      '• $50 off security equipment if cross-selling\\n',
      '• Focus on LIFETIME VALUE increase, not just discounts\\n',
      '• Bundle discounts for multiple services\\n\\n',
      
      '**CROSS-SELL PRIORITY (only if they solve customer problems):**\\n',
      '• Data overages + no unlimited = Unlimited plan upgrade\\n',
      '• WFH + no internet/slow internet = Internet bundle\\n',
      '• High crime area + no security = Home protection\\n',
      '• Multiple family members + individual plans = Family bundle\\n\\n',
      
      '**TONE & APPROACH:**\\n',
      '• Consultative, not pushy\\n',
      '• Problem-solver mindset\\n',
      '• Use "Let me help you save money" not "Let me sell you"\\n',
      '• Reference specific pain points they mentioned\\n',
      '• Show genuine care for their experience\\n\\n',
      
      '**SCRIPT LENGTH:** 200-400 words\\n',
      '**CONVERSATION STYLE:** Natural, empathetic, solution-focused\\n',
      '**INCLUDE:** Specific talking points, transition phrases, objection handling\\n',
      
      'Generate the LIVE CALL personalized retention script the agent can follow with specific talking points:'
    ),
    connection_id => 'us.vertex-ai-connection',
    endpoint => 'gemini-2.5-flash'
  ).result AS live_retention_script

FROM customer_consolidated cc
LEFT JOIN customer_problems cp ON cc.customer_id = cp.customer_id
        """
        
        return query
      

    
    def get_customer_info(self, customer_id):
      """
      Generates the basic details about the customer and initial approaches agent should follow
      """
      query = f"""
      WITH base AS (
        SELECT  
          customer_id,
          age,
          education_level,
          life_stage,
          income_bracket,
          neighborhood_crime_rate,
          work_from_home_flag
        FROM `{self.project_id}.{self.dataset_id}.customer_df`
        WHERE customer_id = '{customer_id}'
      ),

      -- Security services aggregation
      customer_security AS (
        SELECT 
          customer_id,
          STRING_AGG(device_type, ', ' ORDER BY device_type) AS device_types_concat,
          COUNT(device_type) AS total_security_devices
        FROM `{self.project_id}.{self.dataset_id}.security_df`
        WHERE customer_id = '{customer_id}'
        GROUP BY customer_id
      ),

      -- Active internet services
      customer_internet AS (
        SELECT 
          customer_id,
          speed_mbps AS internet_speed,
          plan_tier AS internet_plan,
          contract_type AS internet_contract_type,
          internet_tenure_days
        FROM `{self.project_id}.{self.dataset_id}.internet_df`
        WHERE customer_id = '{customer_id}'
          AND internet_churn = 0
      ),

      -- Active mobile services (non-churned)
      customer_mobile AS (
        SELECT 
          customer_id,
          plan_type AS current_mobile_plan,
          contract_type AS current_mobile_contract_type,
          data_overage_frequency AS current_mobile_data_overage,
          monthly_cost AS current_mobile_cost,
          mobile_tenure_days
        FROM `{self.project_id}.{self.dataset_id}.mobile_df`
        WHERE customer_id = '{customer_id}'
          AND mobile_churn = 0
      ),

      -- Consolidated customer data
      customer_consolidated AS (
        SELECT 
          cb.*,
          
          -- Security services
          COALESCE(cs.device_types_concat, 'None') AS device_types_concat,
          COALESCE(cs.total_security_devices, 0) AS total_security_devices,
          CASE WHEN cs.customer_id IS NOT NULL THEN 1 ELSE 0 END AS security_customer,
          
          -- Internet services
          ci.internet_speed,
          ci.internet_plan,
          ci.internet_contract_type,
          ci.internet_tenure_days,
          CASE 
            WHEN ci.customer_id IS NULL THEN 'No internet service'
            ELSE 'Active internet customer' 
          END AS internet_customer_status,
          
          -- Mobile services (from churn table vs active table)
          cm.current_mobile_plan,
          cm.current_mobile_contract_type,
          cm.current_mobile_data_overage,
          cm.current_mobile_cost,
          cm.mobile_tenure_days,
          CASE 
            WHEN cm.customer_id IS NULL THEN 'Mobile service at risk'
            ELSE 'Active mobile customer'
          END AS mobile_customer_status
          
        FROM base cb
        LEFT JOIN customer_security cs ON cb.customer_id = cs.customer_id
        LEFT JOIN customer_internet ci ON cb.customer_id = ci.customer_id
        LEFT JOIN customer_mobile cm ON cb.customer_id = cm.customer_id
      )

      -- Final output with retention script generation
      SELECT
        AI.GENERATE(
          CONCAT(
            'Create a concise customer summary for a live agent. Format as a professional brief that helps the agent understand who they\\'re speaking with and how to approach the conversation.\\n\\n',
            
            '=== CUSTOMER PROFILE ===\\n',
            'Customer ID: ', customer_id, '\\n',
            'Demographics: ', CAST(age AS STRING), '-year-old, ', education_level, ', ', life_stage, '\\n',
            'Income Level: ', income_bracket, '\\n',
            'Work Setup: ', CASE WHEN work_from_home_flag THEN 'Works from home' ELSE 'Traditional workplace' END, '\\n',
            'Neighborhood: Crime rate ', CAST(neighborhood_crime_rate AS STRING), '/10\\n\\n',
            
            '=== CURRENT SERVICES ===\\n',
            'Mobile: ', mobile_customer_status, 
            CASE WHEN mobile_customer_status = 'Active mobile customer' THEN
              CONCAT(' (', current_mobile_plan, ', $', CAST(current_mobile_cost AS STRING), '/month, ', 
                    CAST(mobile_tenure_days AS STRING), ' days)')
              ELSE '' END, '\\n',
            'Internet: ', internet_customer_status,
            CASE WHEN internet_customer_status = 'Active internet customer' THEN
              CONCAT(' (', internet_plan, ', ', CAST(internet_speed AS STRING), ' Mbps, ', 
                    CAST(internet_tenure_days AS STRING), ' days)')
              ELSE '' END, '\\n',
            'Security: ', CASE WHEN security_customer = 1 
              THEN CONCAT('Active (', CAST(total_security_devices AS STRING), ' devices)')
              ELSE 'Not subscribed' END, '\\n\\n',

            '=== AVAILABLE SOLUTIONS & UPGRADES ===\\n',
            'MOBILE PLANS (solve data overage issues):\\n',
            '• Unlimited Standard: $70/month (save on overages!)\\n',
            '• Unlimited Premium: $90/month (includes Disney+, hotspot)\\n\\n',
            
            'INTERNET SOLUTIONS (solve WFH/speed issues):\\n',
            '• Standard: 100 Mbps at $70/month\\n',
            '• Premium Gig: 1000 Mbps at $100/month (perfect for WFH)\\n\\n',
            
            'SECURITY OPTIONS (solve safety concerns):\\n',
            '• Smart Security: Doorbell + Camera ($200 + free install)\\n',
            '• Complete Protection: Full monitoring ($30/month + $50 off equipment)\\n',
            '• Starter Kit: Motion + Window sensors ($60)\\n\\n',

            '=== AGENT GUIDANCE ===\\n',
            'Create a brief summary section with:\\n',
            '1. WHO THEY ARE: 2-3 sentences describing this customer\\n',
            '2. CONVERSATION APPROACH: How to start the conversation professionally\\n',
            '3. KEY TALKING POINTS: 3-4 relevant topics to discuss based on their profile\\n',
            '4. SERVICE OPPORTUNITIES: Any logical cross-sell or service optimization opportunities\\n\\n',
            
            'Keep the tone professional but warm. Focus on actionable insights for the agent. ',
            'Avoid assumptions - stick to facts from the data. ',
            'Format for quick reading during a live call.'
          ),
          connection_id => 'us.vertex-ai-connection',
          endpoint => 'gemini-2.5-flash-lite'
        ).result AS customer_info
      FROM customer_consolidated;
      """
      return query
    
    def get_business_solution_recommendations(self, problem_description, top_k=3):
      """
      Finds the closest matching business solutions based on problem description
      using vector similarity search
      
      Args:
          problem_description (str): The business problem to find solutions for
          top_k (int): Number of top matches to return (default: 3)
      
      Returns:
          str: SQL query string ready for execution
      """
      query = f"""
      WITH closest_match AS (
        SELECT 
          vs.base.problem AS matched_problem,
          vs.base.solution AS recommended_solution,
          vs.base.sector AS sector,
          vs.distance
        FROM VECTOR_SEARCH(
          TABLE `{self.project_id}.{self.dataset_id}.business_solution_embeddings`,
          'problem_embedding',
          (
            SELECT ml_generate_embedding_result AS query_embedding
            FROM ML.GENERATE_EMBEDDING(
              MODEL `{self.project_id}.{self.dataset_id}.text_embedding_005_model`,
              (SELECT '{problem_description}' AS content)
            )
          ),
          top_k => {top_k},
          distance_type => 'COSINE'
        ) AS vs
      )
      SELECT 
        matched_problem,
        recommended_solution,
        sector,
        distance
      FROM closest_match
      ORDER BY distance ASC;
      """
      return query
    
    
    def get_diy_solution(self, customer_id):
        """
        Collect the DIY solution generated for the customer and walk them trhough it.
        """
        query = f"""
select a.issue_description,a.device_name,a.hardware_or_software,a.urgency,a.diy_instructions,
case when a.need_on_site_visit=True then 'Technnician will visit site soon' else 'We will troubleshoot this on phone now' end as resolution_mode
from `{self.project_id}.{self.dataset_id}.active_customer_tickets_solutions` a 
inner join `{self.project_id}.{self.dataset_id}.active_customer_tickets` b
on a.ticket_id=b.ticket_id
and b.customer_id='{customer_id}'
;

        """
        return query
    

    def get_tech_reports(self, tech_id,city):
        """
        Returns all the active tickets assigned to technician in the city with reports and issue added in.
        """
        query = f"""
select a.ticket_id,a.device_name,a.hardware_or_software,a.urgency,a.issue_description,a.technician_report,b.city,b.postal_code from `{self.project_id}.{self.dataset_id}.active_customer_tickets_mm` b 
inner join `{self.project_id}.{self.dataset_id}.active_customer_tickets_solutions` a
on a.ticket_id=b.ticket_id
where lower(b.city)='{city}' and b.technician_id='{tech_id}' and a.need_on_site_visit=True;
        """
        return query

    def get_tech_solution(self, question):
        """
        Returns the most similar tech issues and solution to them from vector storage to passed pronlem.
        """
        query = f"""
  SELECT 
    vs.base.problem_summary AS matched_problem,
    vs.base.solution_steps AS recommended_solution,
    vs.base.technician_notes AS technician_notes,
    vs.distance
  FROM VECTOR_SEARCH(
    TABLE `{self.project_id}.{self.dataset_id}.device_problem_solution_embeddings`,
    'problem_embedding',
    (
      SELECT ml_generate_embedding_result AS query_embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{self.project_id}.{self.dataset_id}.text_embedding_005_model`,
        (SELECT {question} AS content)
      )
    ),
    top_k => 5,
    distance_type => 'COSINE'
  ) AS vs
  where vs.distance<=0.3
  ORDER BY vs.distance ASC;

        """
        return query




    def get_churn_score_mobile_query(self, customer_id):
        """
        Generate a simpler query focused on churn analysis without the full retention script.
        Useful for quick customer insights.
        """
        query = f"""
SELECT
  customer_id,
  churn_probability_percent,
  risk_category,
  plan_type,
  monthly_cost,
  tenure_years,
  
  ARRAY_TO_STRING(
    ARRAY(
      SELECT CONCAT(feature, ': ', CAST(ROUND(attribution, 3) AS STRING))
      FROM UNNEST(top_feature_attributions)
      ORDER BY ABS(attribution) DESC
      LIMIT 10
    ), ', '
  ) AS top_risk_factors

FROM (
  SELECT *
  FROM ML.EXPLAIN_PREDICT(
    MODEL `{self.project_id}.{self.dataset_id}.mobile_churn_predictor`,
    (SELECT * FROM `{self.project_id}.{self.dataset_id}.mobile_churn_data`
     WHERE customer_id='{customer_id}'
    )
  )
)
        """
        return query
      
      



# # Usage example:
# if __name__ == "__main__":
#     # Initialize the module
#     retention_query = MobileRetentionQuery()
    
#     # Generate query for specific customer
#     customer_id = "c123456"
#     full_query = retention_query.get_retention_query(customer_id)
    
#     print(f"Generated retention query for customer: {customer_id}")
#     print("Query length:", len(full_query))
    
#     # For quick analysis
#     simple_query = retention_query.get_simple_analysis_query(customer_id)
#     print("\\nSimple analysis query generated")
#     print("Query length:", len(simple_query))