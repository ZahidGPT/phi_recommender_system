import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Create models with explicit Input layers
def create_user_model():
    inputs = tf.keras.layers.Input(shape=(4,))
    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(8, activation='relu')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_policy_model():
    inputs = tf.keras.layers.Input(shape=(4,))
    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(8, activation='relu')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

try:
    # Create new models
    user_model = create_user_model()
    policy_model = create_policy_model()
    
    # Compile with explicit loss function
    user_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.mean_squared_error
    )
    
    policy_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.mean_squared_error
    )
    
    # Save models
    user_model.save('user_model.h5')
    policy_model.save('policy_model.h5')
    
    # Verify loading
    test_user = tf.keras.models.load_model('user_model.h5')
    test_policy = tf.keras.models.load_model('policy_model.h5')
    print("Models saved and verified successfully!")
    
except Exception as e:
    print(f"Error: {e}") 