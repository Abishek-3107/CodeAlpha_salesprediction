import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# LOAD OR CREATE DATASET
# ============================================================

def create_sample_dataset():
    """Create a sample advertising dataset if the real one doesn't exist"""
    print("\n📁 Creating sample dataset...")
    
    # Sample data (commonly used in advertising case studies)
    data = {
        'TV': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8,
               66.1, 214.7, 23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3,
               218.4, 237.4, 13.1, 43.1, 28.5, 12.4, 35.9, 78.9, 45.3, 56.2],
        'Radio': [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6,
                 5.8, 24.0, 35.1, 7.6, 32.9, 47.7, 36.6, 19.3, 45.3, 38.6,
                 54.2, 33.9, 47.2, 23.2, 43.3, 23.9, 16.3, 35.5, 37.9, 22.3],
        'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2,
                     24.2, 58.4, 65.6, 32.7, 32.2, 54.6, 52.5, 18.5, 55.7, 57.1,
                     32.8, 52.7, 34.7, 45.4, 27.1, 15.9, 21.1, 24.2, 34.6, 30.6],
        'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6,
                 11.3, 18.2, 11.5, 9.4, 17.2, 12.9, 12.7, 23.7, 12.5, 16.9,
                 18.9, 21.5, 8.9, 9.5, 11.7, 6.5, 8.2, 13.4, 12.9, 9.8]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("advertising.csv", index=False)
    print(" Sample dataset created successfully as 'advertising.csv'")
    return df

# Check if file exists
if os.path.exists("advertising.csv"):
    try:
        df = pd.read_csv("advertising.csv")
        print("\n Loaded existing 'advertising.csv' file")
    except Exception as e:
        print(f"\n⚠️ Error reading file: {e}")
        print("Creating new dataset instead...")
        df = create_sample_dataset()
else:
    print("\n⚠️ 'advertising.csv' not found!")
    df = create_sample_dataset()

# Show first rows
print("\n📊 First 5 Rows of Dataset:\n")
print(df.head())

# ============================================================
# CHECK DATA
# ============================================================

print("\n📏 Dataset Shape:", df.shape)

print("\n🔍 Missing Values:\n")
print(df.isnull().sum())

# Check for any missing values and handle them
if df.isnull().sum().any():
    print("\n⚠️ Dropping rows with missing values...")
    df = df.dropna()

# Verify required columns exist
required_columns = ['TV', 'Radio', 'Newspaper', 'Sales']
if not all(col in df.columns for col in required_columns):
    print(f"\n❌ Error: Dataset must contain columns: {required_columns}")
    print(f"Available columns: {list(df.columns)}")
    # Try to find alternative column names
    for col in df.columns:
        if col.lower() == 'sales' or col.lower() == 'sale':
            df.rename(columns={col: 'Sales'}, inplace=True)
        elif col.lower() == 'tv':
            df.rename(columns={col: 'TV'}, inplace=True)
        elif col.lower() == 'radio':
            df.rename(columns={col: 'Radio'}, inplace=True)
        elif col.lower() == 'newspaper' or col.lower() == 'news':
            df.rename(columns={col: 'Newspaper'}, inplace=True)
    
    # Check again after renaming
    if not all(col in df.columns for col in required_columns):
        print(f"\n❌ Missing required columns after renaming attempt")
        print("Creating sample dataset instead...")
        df = create_sample_dataset()

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# ============================================================
# SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\n📊 Training set size: {X_train.shape[0]} samples")
print(f"📊 Test set size: {X_test.shape[0]} samples")

# ============================================================
# CREATE MODEL
# ============================================================

model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_test)

# ============================================================
# ACCURACY METRICS
# ============================================================

score = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

print("\n" + "="*60)
print("📈 MODEL PERFORMANCE METRICS")
print("="*60)
print(f"R² Score (Accuracy): {round(score, 4)}")
print(f"Mean Squared Error (MSE): {round(mse, 4)}")
print(f"Root Mean Squared Error (RMSE): {round(rmse, 4)}")
print(f"Mean Absolute Error (MAE): {round(mae, 4)}")

# Interpretation of R² score
if score > 0.9:
    print("✅ Excellent model fit!")
elif score > 0.7:
    print("👍 Good model fit!")
elif score > 0.5:
    print("⚠️ Moderate model fit")
else:
    print("❌ Poor model fit - consider adding more features")

# ============================================================
# COEFFICIENTS
# ============================================================

print("\n" + "="*60)
print("🧮 MODEL EQUATION")
print("="*60)
print(f"Sales = {round(model.intercept_, 4)}", end="")
print(f" + ({round(model.coef_[0], 4)} × TV)", end="")
print(f" + ({round(model.coef_[1], 4)} × Radio)", end="")
print(f" + ({round(model.coef_[2], 4)} × Newspaper)")

print("\n📊 Individual Feature Coefficients:")
print(f"  📺 TV        : {round(model.coef_[0], 4)} (Sales increase per $1,000 spent on TV)")
print(f"  📻 Radio     : {round(model.coef_[1], 4)} (Sales increase per $1,000 spent on Radio)")
print(f"  📰 Newspaper : {round(model.coef_[2], 4)} (Sales increase per $1,000 spent on Newspaper)")
print(f"  🎯 Intercept : {round(model.intercept_, 4)} (Base sales with zero advertising)")

# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test, predictions, alpha=0.6, edgecolors='black', linewidth=0.5, color='blue')
plt.xlabel("Actual Sales", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.title("Actual vs Predicted Sales", fontsize=14, fontweight='bold')

# Perfect fit line
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# Add R² score on plot
plt.text(0.05, 0.95, f'R² = {round(score, 4)}', 
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontsize=10, fontweight='bold')

# Plot 2: Residuals
plt.subplot(1, 3, 2)
residuals = y_test - predictions
plt.scatter(predictions, residuals, alpha=0.6, edgecolors='black', linewidth=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel("Predicted Sales", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.title("Residual Plot", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 3: Feature Importance
plt.subplot(1, 3, 3)
features = ['TV', 'Radio', 'Newspaper']
coefficients = model.coef_
colors = ['red', 'blue', 'green']
plt.bar(features, coefficients, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel("Features", fontsize=12)
plt.ylabel("Coefficient Value", fontsize=12)
plt.title("Feature Impact on Sales", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(coefficients):
    plt.text(i, v + (0.01 if v > 0 else -0.05), f'{v:.3f}', 
             ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# CUSTOM PREDICTION
# ============================================================

print("\n" + "="*60)
print("🎯 CUSTOM PREDICTION EXAMPLE")
print("="*60)

# Example 1: Sample from dataset
sample = pd.DataFrame({
    'TV': [230.1],
    'Radio': [37.8],
    'Newspaper': [69.2]
})

predicted_sales = model.predict(sample)
print(f"\n📺 Example 1 - Advertising Budget:")
print(f"  TV        : ${sample['TV'][0]:.1f}k")
print(f"  Radio     : ${sample['Radio'][0]:.1f}k")
print(f"  Newspaper : ${sample['Newspaper'][0]:.1f}k")
print(f"  📈 Predicted Sales: {round(predicted_sales[0], 2)} units")

# Example 2: Custom budget input
print("\n" + "-"*40)
print("💡 Try your own budget:")
try:
    tv_budget = float(input("Enter TV advertising budget (in $1000s): "))
    radio_budget = float(input("Enter Radio advertising budget (in $1000s): "))
    newspaper_budget = float(input("Enter Newspaper advertising budget (in $1000s): "))

    custom_sample = pd.DataFrame({
        'TV': [tv_budget],
        'Radio': [radio_budget],
        'Newspaper': [newspaper_budget]
    })

    custom_prediction = model.predict(custom_sample)
    print(f"\n✨ Predicted Sales: {round(custom_prediction[0], 2)} units")
except ValueError:
    print("⚠️ Invalid input! Please enter numeric values.")

# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================

print("\n" + "="*60)
print("⭐ FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Normalize coefficients for comparison
coef_df = pd.DataFrame({
    'Feature': ['TV', 'Radio', 'Newspaper'],
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
})
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
coef_df['Importance_%'] = (coef_df['Abs_Coefficient'] / coef_df['Abs_Coefficient'].sum() * 100).round(2)

print("\n📊 Feature Impact on Sales (Ranked):")
for idx, row in coef_df.iterrows():
    icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉"
    print(f"  {icon} {row['Feature']:10s}: {row['Coefficient']:+.4f}  (Importance: {row['Importance_%']:.1f}%)")

# ============================================================
# PREDICTIONS ON TEST DATA
# ============================================================

print("\n" + "="*60)
print("🔮 SAMPLE PREDICTIONS (Test Data)")
print("="*60)

comparison_df = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': predictions,
    'Difference': y_test.values - predictions,
    'Absolute Error': np.abs(y_test.values - predictions)
})
comparison_df = comparison_df.head(10)

print("\nFirst 10 Test Set Predictions:")
print(comparison_df.round(2))

# ============================================================
# SAVE MODEL COEFFICIENTS (Optional)
# ============================================================

# Save coefficients to a file for future use
coef_df.to_csv("model_coefficients.csv", index=False)
print("\n💾 Model coefficients saved to 'model_coefficients.csv'")

# ============================================================
# END
# ============================================================

print("\n" + "="*60)
print("✅ SALES PREDICTION COMPLETED SUCCESSFULLY!")
print("="*60)