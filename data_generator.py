import json
import random
import numpy as np

def generate_dummy_data():
    # Locations in Australia
    locations = ['VIC', 'NSW', 'QLD', 'WA', 'SA', 'TAS', 'NT', 'ACT']
    
    # Possible coverage options
    coverage_options = [
        'maternity', 'dental', 'optical', 'hospital', 'neonatal',
        'hospital_partnership', 'physio', 'chiro', 'ambulance',
        'mental_health', 'rehabilitation', 'cardiology'
    ]
    
    # Generate 100 user profiles (increased from 10)
    users = []
    for i in range(100):
        # Age normalized between 18-75 years
        age = round((random.uniform(18, 75) - 18) / 57, 2)
        
        # Budget normalized between 100-1000 AUD
        budget = round((random.uniform(100, 1000) - 100) / 900, 2)
        
        user = {
            "user_id": i,
            "age": age,
            "location": random.choice(locations),
            "maternity_coverage": random.choice([0, 1]),
            "budget": budget,
            "budget_amount": round(budget * 900 + 100, 2),
            "insurer_rating_preference": random.choice([0, 1])
        }
        users.append(user)
    
    # Generate 200 policies (increased from 50)
    policies = []
    for i in range(200):
        num_coverages = random.randint(3, 8)
        selected_coverages = random.sample(coverage_options, num_coverages)
        
        # Premium between 100-1000 AUD
        premium = round(random.uniform(100, 1000) / 1000, 3)
        
        policy = {
            "policy_id": i,
            "premium": premium,
            "premium_amount": round(premium * 1000, 2),
            "waiting_period": random.choice([0, 14, 30, 60, 90, 180]),
            "coverage": selected_coverages,
            "insurer_rating": round(random.uniform(0.7, 0.98), 2)
        }
        policies.append(policy)
    
    # Generate interactions (new)
    interactions = []
    for user in users:
        # Each user interacts with 5-10 policies
        num_interactions = random.randint(5, 10)
        
        # Select random policies that match user preferences
        matching_policies = [
            p for p in policies 
            if p['premium_amount'] <= user['budget_amount'] and
            (not user['maternity_coverage'] or 'maternity' in p['coverage'])
        ]
        
        if matching_policies:
            selected_policies = random.sample(
                matching_policies, 
                min(num_interactions, len(matching_policies))
            )
            
            for policy in selected_policies:
                interaction = {
                    "user_id": user["user_id"],
                    "policy_id": policy["policy_id"],
                    "rating": random.uniform(0.6, 1.0) if random.random() > 0.2 else random.uniform(0.1, 0.5)
                }
                interactions.append(interaction)
    
    return users, policies, interactions

def save_to_json(users, policies, interactions):
    """Save generated data to JSON files"""
    with open('dummy_users.json', 'w') as f:
        json.dump(users, f, indent=4)
        print("Users data saved to dummy_users.json")
    
    with open('dummy_policies.json', 'w') as f:
        json.dump(policies, f, indent=4)
        print("Policies data saved to dummy_policies.json")
    
    with open('dummy_interactions.json', 'w') as f:
        json.dump(interactions, f, indent=4)
        print("Interactions data saved to dummy_interactions.json")

if __name__ == "__main__":
    print("Generating dummy data...")
    users, policies, interactions = generate_dummy_data()
    save_to_json(users, policies, interactions)
    
    # Print statistics
    print("\nData Statistics:")
    print(f"Number of users: {len(users)}")
    print(f"Number of policies: {len(policies)}")
    print(f"Number of interactions: {len(interactions)}")
    
    # Print sample data
    print("\nSample User:")
    print(json.dumps(users[0], indent=2))
    print("\nSample Policy:")
    print(json.dumps(policies[0], indent=2))
    print("\nSample Interaction:")
    print(json.dumps(interactions[0], indent=2)) 