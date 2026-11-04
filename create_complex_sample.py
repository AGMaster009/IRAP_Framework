import pandas as pd
import numpy as np

def generate_realistic_medical_data(num_samples=200):
    np.random.seed(42)
    
    # Generate base features
    patient_ids = np.arange(1, num_samples + 1)
    ages = np.random.randint(20, 85, size=num_samples)
    
    # BMI has a base distribution, with some relation to age
    bmis = np.random.normal(loc=25, scale=5, size=num_samples) + (ages / 50.0)
    
    # Blood Pressure is strongly correlated with Age and BMI
    blood_pressures = 80 + (ages * 0.5) + (bmis * 1.2) + np.random.normal(loc=0, scale=10, size=num_samples)
    
    # Cholesterol correlated with BMI and Age
    cholesterols = 120 + (ages * 1.5) + (bmis * 2.0) + np.random.normal(loc=0, scale=20, size=num_samples)
    
    # Categorical: Low Cardinality
    blood_types = np.random.choice(['A', 'B', 'AB', 'O'], size=num_samples)
    
    # Categorical: High Cardinality (simulating medical codes like ICD-10) -> Should trigger Frequency/Target Encoding
    medical_codes = [f"ICD10_{np.random.randint(100, 150)}" for _ in range(num_samples)]
    
    # Target Variable: Diabetes Risk (0 or 1) based on age and bmi
    risk_score = (ages / 100) + (bmis / 40) - 1.0 + np.random.normal(loc=0, scale=0.2, size=num_samples)
    diabetes_targets = (risk_score > 0).astype(int)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'bmi': bmis,
        'blood_pressure': blood_pressures,
        'cholesterol': cholesterols,
        'blood_type': blood_types,
        'medical_code': medical_codes,
        'diabetes_target': diabetes_targets
    })
    
    # Inject Missing Values (to trigger ContextAwareKNN vs Mean Imputation)
    # 1. Missing BMI (approx 15%)
    bmi_missing_idx = np.random.choice(df.index, size=int(num_samples * 0.15), replace=False)
    df.loc[bmi_missing_idx, 'bmi'] = np.nan
    
    # 2. Missing Blood Pressure (approx 20%)
    bp_missing_idx = np.random.choice(df.index, size=int(num_samples * 0.20), replace=False)
    df.loc[bp_missing_idx, 'blood_pressure'] = np.nan
    
    # 3. Missing Blood Type (Categorical missing)
    bt_missing_idx = np.random.choice(df.index, size=int(num_samples * 0.10), replace=False)
    df.loc[bt_missing_idx, 'blood_type'] = np.nan
    
    # Inject Severe Outliers (to trigger RobustScaler instead of MinMaxScaler)
    outlier_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_idx, 'cholesterol'] = np.random.uniform(800, 1500, size=5) # Impossible biological values
    
    df.to_csv('datasets/realistic_sample.csv', index=False)
    print("Realistic dataset created at datasets/realistic_sample.csv with shape:", df.shape)

if __name__ == "__main__":
    generate_realistic_medical_data()
