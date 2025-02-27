import json
from recommender import HealthInsuranceRecommender
from explanation_generator import generate_explanation
import os

def load_data():
    """Load existing user and policy data from JSON files"""
    try:
        with open('dummy_users.json', 'r') as f:
            users = json.load(f)
        with open('dummy_policies.json', 'r') as f:
            policies = json.load(f)
        return users, policies
    except FileNotFoundError:
        print("Error: Required JSON files not found. Please ensure dummy_users.json and dummy_policies.json exist.")
        return None, None

def get_user_input():
    print("\nPlease enter user details:")
    
    # Get age (18-75 years) and normalize it
    age_years = float(input("Enter age (18-75 years): "))
    while age_years < 18 or age_years > 75:
        print("Age must be between 18 and 75 years.")
        age_years = float(input("Enter age (18-75 years): "))
    age = round((age_years - 18) / 57, 2)  # Normalize to 0-1.0
    
    # Get location
    locations = ['VIC', 'NSW', 'QLD', 'WA', 'SA', 'TAS', 'NT', 'ACT']
    print("Available locations:", ", ".join(locations))
    location = input("Enter location (e.g., VIC): ").upper()
    while location not in locations:
        print("Invalid location. Please choose from:", ", ".join(locations))
        location = input("Enter location: ").upper()
    
    # Get maternity coverage preference
    maternity = input("Need maternity coverage? (yes/no): ").lower()
    maternity_coverage = 1 if maternity.startswith('y') else 0
    
    # Get budget (100-1000) and normalize it
    budget_amount = float(input("Enter monthly budget (100-1000 AUD): "))
    while budget_amount < 100 or budget_amount > 1000:
        print("Budget must be between 100 and 1000 AUD.")
        budget_amount = float(input("Enter monthly budget (100-1000 AUD): "))
    budget = round((budget_amount - 100) / 900, 2)  # Normalize to 0-1.0
    
    # Get insurer rating preference
    rating_pref = input("Prefer high-rated insurers? (yes/no): ").lower()
    insurer_rating_preference = 1 if rating_pref.startswith('y') else 0
    
    return {
        "age": age,
        "location": location,
        "maternity_coverage": maternity_coverage,
        "budget": budget,
        "budget_amount": budget_amount,
        "insurer_rating_preference": insurer_rating_preference
    }

def main():
    # Load existing data
    users, policies = load_data()
    if users is None or policies is None:
        return
    
    # Initialize recommender
    recommender = HealthInsuranceRecommender()
    
    # Debug: Print all files in current directory
    print("\nChecking for model files:")
    for file in os.listdir('.'):
        if 'keras' in file or 'scaler' in file:
            print(f"Found: {file}")
    
    # Check for existing model files with correct names
    model_files = [
        'trained_model.keras.data-00000-of-00001',
        'trained_model.keras.index',
        'user_scaler.save',
        'policy_scaler.save'
    ]
    
    model_exists = all(os.path.exists(f) for f in model_files)
    
    if model_exists:
        print("\nLoading existing trained model...")
        if recommender.load_model():
            recommender.policies = policies
            print("Model loaded successfully!")
        else:
            print("Error loading model. Training new model...")
            recommender.train(users, policies, epochs=50)
    else:
        print("\nMissing model files:", [f for f in model_files if not os.path.exists(f)])
        print("Training new model...")
        recommender.train(users, policies, epochs=50)
    
    # Main recommendation loop
    while True:
        try:
            # Get user input
            user_data = get_user_input()
            
            # Get recommendations
            recommendations = recommender.get_recommendations(user_data)
            
            # Print recommendations
            print("\nPersonalized Health Insurance Recommendations")
            print("-" * 50)
            print(f"User Profile:")
            print(f"Age: {user_data['age']*57+18:.0f} years")
            print(f"Location: {user_data['location']}")
            print(f"Maternity Coverage: {'Yes' if user_data['maternity_coverage'] else 'No'}")
            print(f"Monthly Budget: ${user_data['budget_amount']:.2f}")
            print(f"Prefers High-Rated Insurers: {'Yes' if user_data['insurer_rating_preference'] else 'No'}")
            print("-" * 50)
            
            for i, policy in enumerate(recommendations, 1):
                print(f"\nRecommendation {i}:")
                print(generate_explanation(user_data, policy))
            
            # Ask if user wants to continue
            again = input("\nWould you like another recommendation? (yes/no): ").lower()
            if not again.startswith('y'):
                break
                
        except ValueError as e:
            print("\nError: Please enter valid numerical values.")
            continue
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
            break

if __name__ == "__main__":
    main() 