import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_recommenders as tfrs
from sklearn.preprocessing import StandardScaler
import os
import joblib
import streamlit as st

class HealthInsuranceRecommender(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.user_scaler = StandardScaler()
        self.policy_scaler = StandardScaler()
        self.policies = None
        self.model_path = 'trained_model.keras'
        self.user_scaler_path = 'user_scaler.save'
        self.policy_scaler_path = 'policy_scaler.save'
        
        # Create user tower
        self.user_model = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8)
        ])
        
        # Create policy tower
        self.policy_model = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8)
        ])
        
        # Define task
        self.task = tfrs.tasks.Retrieval(
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=None
            )
        )

    def call(self, features):
        user_embeddings = self.user_model(features["user_features"])
        policy_embeddings = self.policy_model(features["policy_features"])
        return user_embeddings, policy_embeddings

    def compute_loss(self, features, training=False):
        user_embeddings, policy_embeddings = self(features)
        
        # Normalize embeddings
        user_embeddings = tf.nn.l2_normalize(user_embeddings, axis=1)
        policy_embeddings = tf.nn.l2_normalize(policy_embeddings, axis=1)
        
        return self.task(user_embeddings, policy_embeddings, compute_metrics=not training)

    def train(self, users, policies, epochs=50):
        self.policies = policies
        
        # Preprocess features
        user_features, policy_features = self.preprocess_data(users, policies)
        
        # Scale features
        user_features_scaled = self.user_scaler.fit_transform(user_features)
        policy_features_scaled = self.policy_scaler.fit_transform(policy_features)
        
        # Create all possible user-policy pairs
        user_indices = []
        policy_indices = []
        user_features_list = []
        policy_features_list = []
        
        for i in range(len(users)):
            for j in range(len(policies)):
                user_indices.append(i)
                policy_indices.append(j)
                user_features_list.append(user_features_scaled[i])
                policy_features_list.append(policy_features_scaled[j])
        
        # Convert to numpy arrays
        user_features_paired = np.array(user_features_list)
        policy_features_paired = np.array(policy_features_list)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            "user_features": user_features_paired.astype('float32'),
            "policy_features": policy_features_paired.astype('float32')
        }).shuffle(10000).batch(32)
        
        # Set candidates
        policy_features_tensor = tf.convert_to_tensor(
            policy_features_scaled.astype('float32')
        )
        self.task.factorized_metrics.candidates = tf.data.Dataset.from_tensor_slices(
            policy_features_tensor
        ).batch(32)
        
        # Compile
        self.compile(
            optimizer=tf.keras.optimizers.Adam(0.001)
        )
        
        # Train
        history = self.fit(
            dataset,
            epochs=epochs,
            verbose=1
        )
        
        self.save_model()
        return history

    def get_recommendations(self, user_data, k=3):
        """Get top-k policy recommendations for a user"""
        try:
            # Process user data
            user_features = self.preprocess_data([user_data])
            user_features_scaled = self.user_scaler.transform(user_features)
            
            # Get user embedding
            user_embedding = self.user_model(tf.convert_to_tensor(user_features_scaled, dtype=tf.float32))
            user_embedding = tf.nn.l2_normalize(user_embedding, axis=1)
            
            # Get policy embeddings
            policy_features = self.preprocess_data([], self.policies)[1]
            policy_features_scaled = self.policy_scaler.transform(policy_features)
            policy_embeddings = self.policy_model(tf.convert_to_tensor(policy_features_scaled, dtype=tf.float32))
            policy_embeddings = tf.nn.l2_normalize(policy_embeddings, axis=1)
            
            # Calculate base scores using cosine similarity
            base_scores = tf.matmul(user_embedding, tf.transpose(policy_embeddings))[0].numpy()
            
            # Initialize final scores array
            final_scores = np.zeros(len(self.policies))
            
            # Calculate business rule scores for each policy
            for i, policy in enumerate(self.policies):
                score = 50.0  # Start with base score of 50%
                
                # Budget compatibility (±20%)
                if policy['premium_amount'] <= user_data['budget_amount']:
                    score += 20.0
                else:
                    score -= 20.0
                
                # Coverage match (15%)
                if user_data['maternity_coverage'] and 'maternity' in policy['coverage']:
                    score += 15.0
                
                # Insurer rating (10%)
                if user_data['insurer_rating_preference'] and policy['insurer_rating'] > 0.85:
                    score += 10.0
                
                # Waiting period (5%)
                if policy['waiting_period'] <= 30:
                    score += 5.0
                
                # Combine with model score (normalized to ±20%)
                model_contribution = (base_scores[i] + 1) * 10  # Convert from [-1,1] to [0,20]
                
                # Final score
                final_scores[i] = min(max(score + model_contribution, 0), 100)
            
            # Get top-k indices
            top_indices = np.argsort(final_scores)[-k:][::-1]
            
            # Format recommendations
            recommendations = []
            for idx in top_indices:
                policy = self.policies[int(idx)].copy()
                policy['confidence'] = float(final_scores[idx])
                recommendations.append(policy)
            
            return recommendations
            
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return []

    def preprocess_data(self, users, policies=None):
        """Preprocess user and policy data"""
        if users:
            user_features = pd.DataFrame({
                'age': [float(u['age']) for u in users],
                'maternity_coverage': [float(u['maternity_coverage']) for u in users],
                'budget': [float(u['budget']) for u in users],
                'insurer_rating_preference': [float(u['insurer_rating_preference']) for u in users]
            })
        
        if policies is None:
            return user_features
        
        policy_features = pd.DataFrame({
            'premium': [float(p['premium']) for p in policies],
            'waiting_period': [float(p['waiting_period'])/180.0 for p in policies],
            'coverage_count': [float(len(p['coverage']))/12.0 for p in policies],
            'insurer_rating': [float(p['insurer_rating']) for p in policies]
        })
        
        if not users:
            return None, policy_features
            
        return user_features, policy_features

    def save_model(self):
        """Save trained model and scalers"""
        try:
            # Save individual models
            self.user_model.save('user_model.keras')
            self.policy_model.save('policy_model.keras')
            
            # Save scalers
            joblib.dump(self.user_scaler, 'user_scaler.save')
            joblib.dump(self.policy_scaler, 'policy_scaler.save')
            
            print("Model and scalers saved successfully")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """Load trained model and scalers"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Define paths
            user_model_path = os.path.join(current_dir, 'user_model.h5')
            policy_model_path = os.path.join(current_dir, 'policy_model.h5')
            user_scaler_path = os.path.join(current_dir, 'user_scaler.save')
            policy_scaler_path = os.path.join(current_dir, 'policy_scaler.save')
            
            # Load models
            self.user_model = tf.keras.models.load_model(user_model_path)
            self.policy_model = tf.keras.models.load_model(policy_model_path)
            
            # Load scalers
            self.user_scaler = joblib.load(user_scaler_path)
            self.policy_scaler = joblib.load(policy_scaler_path)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False