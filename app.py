import streamlit as st
import json
from recommender import HealthInsuranceRecommender
from explanation_generator import generate_explanation

@st.cache_resource
def load_recommender():
    with st.spinner('Loading recommendation model...'):
        try:
            recommender = HealthInsuranceRecommender()
            if recommender.load_model():
                return recommender
            else:
                st.error("Failed to load model")
                return None
        except Exception as e:
            st.error(f"Error initializing recommender: {str(e)}")
            return None

@st.cache_data
def load_data():
    try:
        with open('dummy_users.json', 'r') as f:
            users = json.load(f)
        with open('dummy_policies.json', 'r') as f:
            policies = json.load(f)
        return users, policies
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def main():
    st.title("Personal Health Insurance Recommender")
    st.subheader("Find the perfect health insurance policy for your needs")
    
    # Load the recommender
    recommender = load_recommender()
    if recommender is None:
        st.error("Could not initialize the recommendation system")
        st.stop()
        return
    
    # Load data
    users, policies = load_data()
    if users is None or policies is None:
        st.error("Could not load necessary data")
        st.stop()
        return
    
    try:
        # Set policies
        recommender.policies = policies
        
        # Create sidebar for user inputs
        st.sidebar.header("Your Preferences")
        
        # Get age input (18-75 years) and normalize it
        age_years = st.sidebar.slider("Age", 18, 75, 30)
        age = round((age_years - 18) / 57, 2)  # Normalize to 0-1.0
        
        # Get location
        locations = ['VIC', 'NSW', 'QLD', 'WA', 'SA', 'TAS', 'NT', 'ACT']
        location = st.sidebar.selectbox("Location", locations)
        
        # Get maternity coverage preference
        maternity_coverage = st.sidebar.checkbox("Need maternity coverage?")
        
        # Get budget (100-1000) and normalize it
        budget_amount = st.sidebar.slider("Monthly Budget (AUD)", 100, 1000, 500)
        budget = round((budget_amount - 100) / 900, 2)  # Normalize to 0-1.0
        
        # Get insurer rating preference
        insurer_rating_preference = st.sidebar.checkbox("Prefer high-rated insurers?")
        
        # Create user data dictionary
        user_data = {
            "age": age,
            "location": location,
            "maternity_coverage": 1 if maternity_coverage else 0,
            "budget": budget,
            "budget_amount": budget_amount,
            "insurer_rating_preference": 1 if insurer_rating_preference else 0
        }
        
        # Add a "Get Recommendations" button
        if st.sidebar.button("Get Recommendations"):
            # Show user profile
            st.subheader("Your Profile")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Age: {age_years} years")
                st.write(f"Location: {location}")
                st.write(f"Maternity Coverage: {'Yes' if maternity_coverage else 'No'}")
            with col2:
                st.write(f"Monthly Budget: ${budget_amount:.2f}")
                st.write(f"Prefers High-Rated Insurers: {'Yes' if insurer_rating_preference else 'No'}")
            
            # Get and display recommendations
            recommendations = recommender.get_recommendations(user_data)
            
            st.subheader("Recommended Policies")
            for i, policy in enumerate(recommendations, 1):
                with st.expander(f"Recommendation {i}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        explanation = generate_explanation(user_data, policy)
                        st.markdown(explanation)
                    
                    with col2:
                        # Visual indicators
                        if policy['premium_amount'] <= budget_amount:
                            st.success("💰 Within Budget!")
                        
                        if policy['insurer_rating'] > 0.85:
                            st.success("⭐ Top-Rated Insurer")
                        
                        if policy['waiting_period'] <= 30:
                            st.info("⚡ Quick Start")
                        
                        if len(policy['coverage']) >= 6:
                            st.info("🛡️ Comprehensive")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main() 