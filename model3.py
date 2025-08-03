import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedTravelRecommendationSystem:
    def __init__(self):
        self.hotel_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=20)
        self.restaurant_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=20)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.df_original = None
        self.hotel_satisfaction_scores = {}
        self.restaurant_satisfaction_scores = {}
        
    def calculate_satisfaction_scores(self, df):
        """Calculate satisfaction percentage for each hotel and restaurant by location"""
        print("Calculating satisfaction scores...")
        
        # Calculate hotel satisfaction scores by location
        hotel_satisfaction = {}
        restaurant_satisfaction = {}
        
        for location in df['travel_place'].unique():
            location_data = df[df['travel_place'] == location]
            
            # Hotel satisfaction scores for this location
            for hotel in location_data['hotel_name'].unique():
                hotel_data = location_data[location_data['hotel_name'] == hotel]
                yes_count = (hotel_data['did_you_like_the_hotel'] == 'yes').sum()
                total_count = len(hotel_data)
                satisfaction_rate = yes_count / total_count if total_count > 0 else 0.5
                hotel_satisfaction[f"{hotel}_{location}"] = satisfaction_rate
            
            # Restaurant satisfaction scores for this location
            for restaurant in location_data['recommended_restaurant'].unique():
                restaurant_data = location_data[location_data['recommended_restaurant'] == restaurant]
                yes_count = (restaurant_data['did_you_like_the_restaurant'] == 'yes').sum()
                total_count = len(restaurant_data)
                satisfaction_rate = yes_count / total_count if total_count > 0 else 0.5
                restaurant_satisfaction[f"{restaurant}_{location}"] = satisfaction_rate
        
        self.hotel_satisfaction_scores = hotel_satisfaction
        self.restaurant_satisfaction_scores = restaurant_satisfaction
        
        print(f"Calculated satisfaction scores for {len(hotel_satisfaction)} hotel-location pairs")
        print(f"Calculated satisfaction scores for {len(restaurant_satisfaction)} restaurant-location pairs")
        
        return hotel_satisfaction, restaurant_satisfaction
    
    def calculate_combined_value_satisfaction_score(self, df, item_column, satisfaction_scores, location_col='travel_place'):
        """Calculate combined value-for-money and satisfaction score"""
        combined_scores = []
        
        for _, row in df.iterrows():
            item_name = row[item_column]
            location = row[location_col]
            budget = (row['min_cost_per_person'] + row['max_cost_per_person']) / 2
            
            # Get location-specific data for value calculation
            location_data = df[df[location_col] == location]
            
            # Calculate value score (inverse of cost - lower cost = higher value)
            max_cost = location_data['max_cost_per_person'].max()
            min_cost = location_data['min_cost_per_person'].min()
            
            if max_cost > min_cost:
                value_score = 1 - ((budget - min_cost) / (max_cost - min_cost))
            else:
                value_score = 0.5
            
            # Get satisfaction score
            item_location_key = f"{item_name}_{location}"
            satisfaction_score = satisfaction_scores.get(item_location_key, 0.5)
            
            # Combine value and satisfaction (weighted combination)
            # Give more weight to satisfaction (60%) and value (40%)
            combined_score = (0.4 * value_score) + (0.6 * satisfaction_score)
            combined_scores.append(combined_score)
        
        return combined_scores
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the enhanced travel data with satisfaction feedback"""
        print("Loading enhanced data with user satisfaction...")
        df = pd.read_csv(csv_path)
        
        # Store original data
        self.df_original = df.copy()
        
        # Handle missing values
        df = df.dropna()
        print(f"Data shape after removing nulls: {df.shape}")
        
        # Calculate satisfaction scores by location
        hotel_satisfaction, restaurant_satisfaction = self.calculate_satisfaction_scores(df)
        
        # Encode categorical features
        categorical_features = ['travel_month', 'travel_place']
        
        for feature in categorical_features:
            le = LabelEncoder()
            df[f'{feature}_encoded'] = le.fit_transform(df[feature])
            self.label_encoders[feature] = le
        
        # Calculate combined value-satisfaction scores
        df['hotel_combined_score'] = self.calculate_combined_value_satisfaction_score(
            df, 'hotel_name', hotel_satisfaction
        )
        df['restaurant_combined_score'] = self.calculate_combined_value_satisfaction_score(
            df, 'recommended_restaurant', restaurant_satisfaction
        )
        
        # Prepare features (the 4 input features)
        feature_columns = ['min_cost_per_person', 'max_cost_per_person', 
                          'travel_month_encoded', 'travel_place_encoded']
        
        X = df[feature_columns]
        y_hotel = df['hotel_combined_score']
        y_restaurant = df['restaurant_combined_score']
        
        return X, y_hotel, y_restaurant, df
    
    def train_models(self, csv_path):
        """Train enhanced hotel and restaurant recommendation models"""
        print("Training enhanced models with satisfaction feedback...")
        
        # Load and preprocess data
        X, y_hotel, y_restaurant, df = self.load_and_preprocess_data(csv_path)
        
        # Split data
        X_train, X_test, y_hotel_train, y_hotel_test = train_test_split(
            X, y_hotel, test_size=0.2, random_state=42
        )
        
        _, _, y_restaurant_train, y_restaurant_test = train_test_split(
            X, y_restaurant, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train hotel model
        print("Training enhanced hotel recommendation model...")
        self.hotel_model.fit(X_train_scaled, y_hotel_train)
        hotel_pred = self.hotel_model.predict(X_test_scaled)
        hotel_mse = mean_squared_error(y_hotel_test, hotel_pred)
        hotel_r2 = r2_score(y_hotel_test, hotel_pred)
        
        # Train restaurant model
        print("Training enhanced restaurant recommendation model...")
        self.restaurant_model.fit(X_train_scaled, y_restaurant_train)
        restaurant_pred = self.restaurant_model.predict(X_test_scaled)
        restaurant_mse = mean_squared_error(y_restaurant_test, restaurant_pred)
        restaurant_r2 = r2_score(y_restaurant_test, restaurant_pred)
        
        print(f"\nEnhanced Hotel Model Performance:")
        print(f"MSE: {hotel_mse:.4f}")
        print(f"R² Score: {hotel_r2:.4f}")
        
        print(f"\nEnhanced Restaurant Model Performance:")
        print(f"MSE: {restaurant_mse:.4f}")
        print(f"R² Score: {restaurant_r2:.4f}")
        
        # Calculate baseline performance
        baseline_hotel_r2, baseline_restaurant_r2 = self.calculate_baseline_performance(
            df, X_test, y_hotel_test, y_restaurant_test
        )
        
        # Create performance visualization
        self.create_performance_visualization(
            hotel_r2, restaurant_r2, baseline_hotel_r2, baseline_restaurant_r2,
            hotel_pred, y_hotel_test, restaurant_pred, y_restaurant_test
        )
        
        # Show satisfaction score examples
        self.show_satisfaction_examples()
        
        return hotel_r2, restaurant_r2, baseline_hotel_r2, baseline_restaurant_r2
    
    def calculate_baseline_performance(self, df, X_test, y_hotel_test, y_restaurant_test):
        hotel_avg_scores = df.groupby('hotel_name')['hotel_combined_score'].mean()
        restaurant_avg_scores = df.groupby('recommended_restaurant')['restaurant_combined_score'].mean()
        
        most_popular_hotel = df['hotel_name'].mode()[0]
        most_popular_restaurant = df['recommended_restaurant'].mode()[0]
        
        hotel_baseline_score = hotel_avg_scores[most_popular_hotel]
        restaurant_baseline_score = restaurant_avg_scores[most_popular_restaurant]
        
        # Create baseline predictions
        hotel_baseline_pred = np.full(len(y_hotel_test), hotel_baseline_score)
        restaurant_baseline_pred = np.full(len(y_restaurant_test), restaurant_baseline_score)
        
        # Calculate R² for baseline
        baseline_hotel_r2 = r2_score(y_hotel_test, hotel_baseline_pred)
        baseline_restaurant_r2 = r2_score(y_restaurant_test, restaurant_baseline_pred)
        
        print(f"\nBaseline Performance (Most Popular with Satisfaction):")
        print(f"Hotel Baseline R²: {baseline_hotel_r2:.4f}")
        print(f"Restaurant Baseline R²: {baseline_restaurant_r2:.4f}")
        
        return baseline_hotel_r2, baseline_restaurant_r2
    
    def show_satisfaction_examples(self):
        """Show examples of satisfaction scores"""
        print(f"\nSample Hotel Satisfaction Scores:")
        hotel_items = list(self.hotel_satisfaction_scores.items())[:5]
        for item, score in hotel_items:
            print(f"{item}: {score:.2%}")
        
        print(f"\nSample Restaurant Satisfaction Scores:")
        restaurant_items = list(self.restaurant_satisfaction_scores.items())[:5]
        for item, score in restaurant_items:
            print(f"{item}: {score:.2%}")
    
    def create_performance_visualization(self, hotel_r2, restaurant_r2, baseline_hotel_r2, baseline_restaurant_r2,
                                       hotel_pred, y_hotel_test, restaurant_pred, y_restaurant_test):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Model vs Baseline comparison
        categories = ['Hotel Model', 'Restaurant Model']
        model_scores = [hotel_r2, restaurant_r2]
        baseline_scores = [baseline_hotel_r2, baseline_restaurant_r2]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, model_scores, width, label='Enhanced RF Model', color='darkgreen')
        axes[0, 0].bar(x + width/2, baseline_scores, width, label='Baseline (Most Popular)', color='lightcoral')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Enhanced Model Performance vs Baseline')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(categories)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hotel predictions scatter plot
        axes[0, 1].scatter(y_hotel_test, hotel_pred, alpha=0.6, color='darkblue')
        axes[0, 1].plot([y_hotel_test.min(), y_hotel_test.max()], [y_hotel_test.min(), y_hotel_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Hotel Combined Scores')
        axes[0, 1].set_ylabel('Predicted Hotel Combined Scores')
        axes[0, 1].set_title(f'Hotel Predictions (R² = {hotel_r2:.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Restaurant predictions scatter plot
        axes[0, 2].scatter(y_restaurant_test, restaurant_pred, alpha=0.6, color='darkgreen')
        axes[0, 2].plot([y_restaurant_test.min(), y_restaurant_test.max()], [y_restaurant_test.min(), y_restaurant_test.max()], 'r--', lw=2)
        axes[0, 2].set_xlabel('Actual Restaurant Combined Scores')
        axes[0, 2].set_ylabel('Predicted Restaurant Combined Scores')
        axes[0, 2].set_title(f'Restaurant Predictions (R² = {restaurant_r2:.4f})')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Feature importance
        feature_names = ['Min Cost', 'Max Cost', 'Travel Month', 'Travel Place']
        hotel_importance = self.hotel_model.feature_importances_
        restaurant_importance = self.restaurant_model.feature_importances_
        
        x_feat = np.arange(len(feature_names))
        axes[1, 0].bar(x_feat - width/2, hotel_importance, width, label='Hotel Model', color='darkblue')
        axes[1, 0].bar(x_feat + width/2, restaurant_importance, width, label='Restaurant Model', color='darkgreen')
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].set_title('Feature Importance Comparison')
        axes[1, 0].set_xticks(x_feat)
        axes[1, 0].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Satisfaction score distribution for hotels
        hotel_sat_scores = list(self.hotel_satisfaction_scores.values())
        axes[1, 1].hist(hotel_sat_scores, bins=20, alpha=0.7, color='darkblue', edgecolor='black')
        axes[1, 1].set_xlabel('Satisfaction Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Hotel Satisfaction Score Distribution')
        axes[1, 1].axvline(np.mean(hotel_sat_scores), color='red', linestyle='--', label=f'Mean: {np.mean(hotel_sat_scores):.2f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Satisfaction score distribution for restaurants
        restaurant_sat_scores = list(self.restaurant_satisfaction_scores.values())
        axes[1, 2].hist(restaurant_sat_scores, bins=20, alpha=0.7, color='darkgreen', edgecolor='black')
        axes[1, 2].set_xlabel('Satisfaction Score')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Restaurant Satisfaction Score Distribution')
        axes[1, 2].axvline(np.mean(restaurant_sat_scores), color='red', linestyle='--', label=f'Mean: {np.mean(restaurant_sat_scores):.2f}')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_model_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print improvement over baseline
        hotel_improvement = ((hotel_r2 - baseline_hotel_r2) / abs(baseline_hotel_r2) * 100) if baseline_hotel_r2 != 0 else float('inf')
        restaurant_improvement = ((restaurant_r2 - baseline_restaurant_r2) / abs(baseline_restaurant_r2) * 100) if baseline_restaurant_r2 != 0 else float('inf')
        
        print(f"\nEnhanced Model Improvements over Baseline:")
        print(f"Hotel Model: {hotel_improvement:.1f}% improvement")
        print(f"Restaurant Model: {restaurant_improvement:.1f}% improvement")
    
    def save_model(self, model_path='enhanced_travel_recommendation_model.pkl'):
        """Save the enhanced trained model"""
        model_data = {
            'hotel_model': self.hotel_model,
            'restaurant_model': self.restaurant_model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'df_original': self.df_original,
            'hotel_satisfaction_scores': self.hotel_satisfaction_scores,
            'restaurant_satisfaction_scores': self.restaurant_satisfaction_scores
        }
        
        joblib.dump(model_data, model_path)
        print(f"Enhanced model saved successfully as {model_path}")
    
    def load_model(self, model_path='enhanced_travel_recommendation_model.pkl'):
        """Load the enhanced pre-trained model"""
        try:
            model_data = joblib.load(model_path)
            self.hotel_model = model_data['hotel_model']
            self.restaurant_model = model_data['restaurant_model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.df_original = model_data['df_original']
            self.hotel_satisfaction_scores = model_data['hotel_satisfaction_scores']
            self.restaurant_satisfaction_scores = model_data['restaurant_satisfaction_scores']
            print("Enhanced model loaded successfully!")
            return True
        except FileNotFoundError:
            print("Enhanced model file not found. Please train the model first.")
            return False

# Main training and demonstration function
def main():
    print("="*80)
    print("ENHANCED TRAVEL RECOMMENDATION SYSTEM WITH USER SATISFACTION")
    print("="*80)
    
    # Initialize the enhanced system
    recommender = EnhancedTravelRecommendationSystem()
    
    # Train the models
    csv_path = 'updated_travel_data_1000.csv'
    
    try:
        hotel_r2, restaurant_r2, baseline_hotel_r2, baseline_restaurant_r2 = recommender.train_models(csv_path)
        
        # Save the enhanced model
        recommender.save_model()
        
        print("\n" + "="*80)
        print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Enhanced Hotel Model R² Score: {hotel_r2:.4f}")
        print(f"Enhanced Restaurant Model R² Score: {restaurant_r2:.4f}")
        print(f"Model saved as 'enhanced_travel_recommendation_model.pkl'")
        print("Performance visualization saved as 'enhanced_model_performance_metrics.png'")
        
    except Exception as e:
        print(f"Error during enhanced training: {e}")
        print("Please ensure 'updated_travel_data_1000.csv' exists and is properly formatted.")

if __name__ == "__main__":
    main()