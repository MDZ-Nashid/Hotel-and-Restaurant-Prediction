import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedTravelRecommendationPredictor:
    def __init__(self, model_path='main_model.pkl'):
        """Initialize the enhanced predictor and load the trained model"""
        self.hotel_model = None
        self.restaurant_model = None
        self.label_encoders = {}
        self.scaler = None
        self.df_original = None
        self.hotel_satisfaction_scores = {}
        self.restaurant_satisfaction_scores = {}
        
        # Load the model automatically on initialization
        self.load_model(model_path)
    
    def load_model(self, model_path='main_model.pkl'):
        """Load the enhanced pre-trained model and all necessary components"""
        try:
            print("Loading enhanced trained model with satisfaction data...")
            model_data = joblib.load(model_path)
            
            self.hotel_model = model_data['hotel_model']
            self.restaurant_model = model_data['restaurant_model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.df_original = model_data['df_original']
            self.hotel_satisfaction_scores = model_data['hotel_satisfaction_scores']
            self.restaurant_satisfaction_scores = model_data['restaurant_satisfaction_scores']
            

            return True
            
        except FileNotFoundError:
            print(f"Enhanced model file '{model_path}' not found.")
            print("Please ensure you have trained the enhanced model first using the training script.")
            return False
        except Exception as e:
            print(f"Error loading enhanced model: {e}")
            return False
    
    def get_recommendations(self, min_cost, max_cost, travel_month, travel_place):
        """Get top 3 hotel and restaurant recommendations with strict budget allocation"""
        if self.hotel_model is None:
            print("Enhanced model not loaded. Please check if the model file exists.")
            return None
            
        try:
            # Validate inputs
            if travel_month not in self.label_encoders['travel_month'].classes_:
                print(f"Available months: {list(self.label_encoders['travel_month'].classes_)}")
                return None
                
            if travel_place not in self.label_encoders['travel_place'].classes_:
                print(f"Available places: {list(self.label_encoders['travel_place'].classes_)}")
                return None
            
            # Budget allocation: 50% for hotel, 50% for restaurant
            hotel_max_budget = max_cost * 0.5
            restaurant_max_budget = max_cost * 0.5
            hotel_min_budget = min_cost * 0.5
            restaurant_min_budget = min_cost * 0.5
            
            print(f"Budget allocation - Hotel: {hotel_min_budget:.0f}-{hotel_max_budget:.0f} BDT, Restaurant: {restaurant_min_budget:.0f}-{restaurant_max_budget:.0f} BDT")
            
            # Encode categorical inputs
            travel_month_encoded = self.label_encoders['travel_month'].transform([travel_month])[0]
            travel_place_encoded = self.label_encoders['travel_place'].transform([travel_place])[0]
            
            # Filter data for specific location
            location_data = self.df_original[self.df_original['travel_place'] == travel_place]
            
            if len(location_data) == 0:
                print(f"Location '{travel_place}' not found in dataset.")
                return None
            
            # Filter hotels within hotel budget allocation
            hotel_budget_mask = (
                (location_data['min_cost_per_person'] <= hotel_max_budget) & 
                (location_data['max_cost_per_person'] >= hotel_min_budget)
            )
            hotel_filtered_df = location_data[hotel_budget_mask]
            
            # Filter restaurants within restaurant budget allocation
            restaurant_budget_mask = (
                (location_data['min_cost_per_person'] <= restaurant_max_budget) & 
                (location_data['max_cost_per_person'] >= restaurant_min_budget)
            )
            restaurant_filtered_df = location_data[restaurant_budget_mask]
            
            # Check if we have options for both hotels and restaurants
            if len(hotel_filtered_df) == 0:
                print(f"No hotels found in {travel_place} within hotel budget ({hotel_min_budget:.0f}-{hotel_max_budget:.0f} BDT).")
                min_hotel_cost = location_data['min_cost_per_person'].min()
                max_hotel_cost = location_data['max_cost_per_person'].max()
                print(f"Available hotel cost range in {travel_place}: {min_hotel_cost}-{max_hotel_cost} BDT")
                return None
                
            if len(restaurant_filtered_df) == 0:
                print(f"No restaurants found in {travel_place} within restaurant budget ({restaurant_min_budget:.0f}-{restaurant_max_budget:.0f} BDT).")
                min_restaurant_cost = location_data['min_cost_per_person'].min()
                max_restaurant_cost = location_data['max_cost_per_person'].max()
                print(f"Available restaurant cost range in {travel_place}: {min_restaurant_cost}-{max_restaurant_cost} BDT")
                return None
            
            print(f"Found {len(hotel_filtered_df['hotel_name'].unique())} hotels and {len(restaurant_filtered_df['recommended_restaurant'].unique())} restaurants in {travel_place} within allocated budgets")
            
            # Get unique hotels and restaurants within their respective budgets
            budget_hotels = hotel_filtered_df['hotel_name'].unique()
            budget_restaurants = restaurant_filtered_df['recommended_restaurant'].unique()
            
            # Prepare input for prediction
            input_features = np.array([[min_cost, max_cost, travel_month_encoded, travel_place_encoded]])
            input_scaled = self.scaler.transform(input_features)
            
            # Get model predictions
            hotel_pred_score = self.hotel_model.predict(input_scaled)[0]
            restaurant_pred_score = self.restaurant_model.predict(input_scaled)[0]
            
            # Score hotels within budget allocation
            hotel_scores = {}
            
            # Calculate hotel cost range for value scoring
            hotel_min_cost = hotel_filtered_df['min_cost_per_person'].min()
            hotel_max_cost = hotel_filtered_df['max_cost_per_person'].max()
            
            for hotel in budget_hotels:
                hotel_data = hotel_filtered_df[hotel_filtered_df['hotel_name'] == hotel]
                
                # Calculate average cost for this hotel
                avg_cost = (hotel_data['min_cost_per_person'].mean() + hotel_data['max_cost_per_person'].mean()) / 2
                
                # STRICT BUDGET CHECK: Skip if average cost exceeds allocated budget
                if avg_cost > hotel_max_budget:
                    continue
                
                # Get satisfaction score for this hotel-location combination
                hotel_location_key = f"{hotel}_{travel_place}"
                satisfaction_score = self.hotel_satisfaction_scores.get(hotel_location_key, 0.5)
                
                # Calculate value score within hotel budget range
                if hotel_max_cost > hotel_min_cost:
                    value_score = 1 - ((avg_cost - hotel_min_cost) / (hotel_max_cost - hotel_min_cost))
                else:
                    value_score = 0.5
                
                # Combine value, satisfaction, and model prediction
                combined_score = (0.25 * value_score) + (0.40 * satisfaction_score) + (0.35 * hotel_pred_score)
                
                hotel_scores[hotel] = {
                    'combined_score': combined_score,
                    'satisfaction_rate': satisfaction_score,
                    'avg_cost': avg_cost,
                    'value_score': value_score,
                    'model_score': hotel_pred_score
                }
            
            # Score restaurants within budget allocation
            restaurant_scores = {}
            
            # Calculate restaurant cost range for value scoring
            restaurant_min_cost = restaurant_filtered_df['min_cost_per_person'].min()
            restaurant_max_cost = restaurant_filtered_df['max_cost_per_person'].max()
            
            for restaurant in budget_restaurants:
                restaurant_data = restaurant_filtered_df[restaurant_filtered_df['recommended_restaurant'] == restaurant]
                
                # Calculate average cost for this restaurant
                avg_cost = (restaurant_data['min_cost_per_person'].mean() + restaurant_data['max_cost_per_person'].mean()) / 2
                
                # STRICT BUDGET CHECK: Skip if average cost exceeds allocated budget
                if avg_cost > restaurant_max_budget:
                    continue
                
                # Get satisfaction score for this restaurant-location combination
                restaurant_location_key = f"{restaurant}_{travel_place}"
                satisfaction_score = self.restaurant_satisfaction_scores.get(restaurant_location_key, 0.5)
                
                # Calculate value score within restaurant budget range
                if restaurant_max_cost > restaurant_min_cost:
                    value_score = 1 - ((avg_cost - restaurant_min_cost) / (restaurant_max_cost - restaurant_min_cost))
                else:
                    value_score = 0.5
                
                # Combine value, satisfaction, and model prediction
                combined_score = (0.25 * value_score) + (0.40 * satisfaction_score) + (0.35 * restaurant_pred_score)
                
                restaurant_scores[restaurant] = {
                    'combined_score': combined_score,
                    'satisfaction_rate': satisfaction_score,
                    'avg_cost': avg_cost,
                    'value_score': value_score,
                    'model_score': restaurant_pred_score
                }
            
            # Check if we have any options after strict budget filtering
            if len(hotel_scores) == 0:
                print(f"No hotels in {travel_place} have average costs within the allocated hotel budget ({hotel_max_budget:.0f} BDT).")
                return None
                
            if len(restaurant_scores) == 0:
                print(f"No restaurants in {travel_place} have average costs within the allocated restaurant budget ({restaurant_max_budget:.0f} BDT).")
                return None
            
            # Get top 3 recommendations prioritized by satisfaction within budget
            top_3_hotels = sorted(hotel_scores.items(), 
                                key=lambda x: (x[1]['satisfaction_rate'], x[1]['combined_score']), 
                                reverse=True)[:3]
            top_3_restaurants = sorted(restaurant_scores.items(), 
                                     key=lambda x: (x[1]['satisfaction_rate'], x[1]['combined_score']), 
                                     reverse=True)[:3]
            
            return {
                'hotels': top_3_hotels,
                'restaurants': top_3_restaurants,
                'total_hotel_options': len(hotel_scores),
                'total_restaurant_options': len(restaurant_scores),
                'budget_allocation': {
                    'hotel_budget': f"{hotel_min_budget:.0f}-{hotel_max_budget:.0f} BDT",
                    'restaurant_budget': f"{restaurant_min_budget:.0f}-{restaurant_max_budget:.0f} BDT"
                },
                'input_params': {
                    'min_cost': min_cost,
                    'max_cost': max_cost,
                    'travel_month': travel_month,
                    'travel_place': travel_place
                }
            }
            
        except Exception as e:
            print(f"Error making enhanced recommendations: {e}")
            return None
    
    def display_recommendations(self, recommendations):
        if not recommendations:
            print("No recommendations to display.")
            return
        
        params = recommendations['input_params']
        budget_allocation = recommendations['budget_allocation']
        
        print("TRAVEL RECOMMENDATIONS WITH BUDGET ALLOCATION")
        print("="*70)
        print(f"Total Budget: {params['min_cost']}-{params['max_cost']} BDT")
        print(f"Hotel Budget: {budget_allocation['hotel_budget']}")
        print(f"Restaurant Budget: {budget_allocation['restaurant_budget']}")
        print(f"Month: {params['travel_month']}")
        print(f"Destination: {params['travel_place']}")
        print(f"Available options: {recommendations['total_hotel_options']} hotels, {recommendations['total_restaurant_options']} restaurants")
        
        print(f"\n TOP 3 HOTEL RECOMMENDATIONS:")
        print("-" * 50)
        for i, (hotel, scores) in enumerate(recommendations['hotels'], 1):
            print(f"{i}. {hotel}")
            print(f" User Satisfaction: {scores['satisfaction_rate']:.1%}")
            print(f"Combined Score: {scores['combined_score']:.4f}")
            print(f"Average Cost: {scores['avg_cost']:.0f} BDT")
            print(f"Value Score: {scores['value_score']:.3f}")
            print()
        
        print(f"TOP 3 RESTAURANT RECOMMENDATIONS:")
        print("-" * 50)
        for i, (restaurant, scores) in enumerate(recommendations['restaurants'], 1):
            print(f"{i}. {restaurant}")
            print(f" User Satisfaction: {scores['satisfaction_rate']:.1%}")
            print(f" Combined Score: {scores['combined_score']:.4f}")
            print(f" Average Cost: {scores['avg_cost']:.0f} BDT")
            print(f" Value Score: {scores['value_score']:.3f}")
            print()
        
        if recommendations['hotels'] and recommendations['restaurants']:
            total_min_cost = recommendations['hotels'][0][1]['avg_cost'] + recommendations['restaurants'][0][1]['avg_cost']
            print(f"Estimated Total Cost (Top Recommendations): {total_min_cost:.0f} BDT")
            if total_min_cost <= params['max_cost']:
                print("Total cost is within your budget!")
            else:
                print("Total cost might exceed your budget. Consider adjusting selections.")
    
    def get_satisfaction_stats(self, travel_place=None):
        if travel_place:
            # Filter by location
            hotel_stats = {k: v for k, v in self.hotel_satisfaction_scores.items() if k.endswith(f"_{travel_place}")}
            restaurant_stats = {k: v for k, v in self.restaurant_satisfaction_scores.items() if k.endswith(f"_{travel_place}")}
            print(f"\nSatisfaction Statistics for {travel_place}:")
        else:
            hotel_stats = self.hotel_satisfaction_scores
            restaurant_stats = self.restaurant_satisfaction_scores
            print(f"\nOverall Satisfaction Statistics:")
        
        if hotel_stats:
            hotel_scores = list(hotel_stats.values())
            print(f"Hotels - Average Satisfaction: {np.mean(hotel_scores):.1%}")
            print(f"Hotels - Highest Satisfaction: {max(hotel_scores):.1%}")
            print(f"Hotels - Lowest Satisfaction: {min(hotel_scores):.1%}")
        
        if restaurant_stats:
            restaurant_scores = list(restaurant_stats.values())
            print(f"Restaurants - Average Satisfaction: {np.mean(restaurant_scores):.1%}")
            print(f"Restaurants - Highest Satisfaction: {max(restaurant_scores):.1%}")
            print(f"Restaurants - Lowest Satisfaction: {min(restaurant_scores):.1%}")
    
    def get_available_options(self):
        """Get all available travel months and places with satisfaction info"""
        if self.label_encoders:
            return {
                'travel_months': list(self.label_encoders['travel_month'].classes_),
                'travel_places': list(self.label_encoders['travel_place'].classes_),
                'budget_info': {
                    'min_cost': int(self.df_original['min_cost_per_person'].min()),
                    'max_cost': int(self.df_original['max_cost_per_person'].max()),
                    'avg_cost': int(self.df_original[['min_cost_per_person', 'max_cost_per_person']].mean().mean())
                },
                'satisfaction_info': {
                    'total_hotel_ratings': len(self.hotel_satisfaction_scores),
                    'total_restaurant_ratings': len(self.restaurant_satisfaction_scores),
                    'avg_hotel_satisfaction': f"{np.mean(list(self.hotel_satisfaction_scores.values())):.1%}",
                    'avg_restaurant_satisfaction': f"{np.mean(list(self.restaurant_satisfaction_scores.values())):.1%}"
                }
            }
        return None

# Convenience functions for easy usage
def make_enhanced_prediction(min_cost, max_cost, travel_month, travel_place, 
                           model_path='main_model.pkl'):
    """Quick function to make an enhanced prediction with satisfaction scores"""
    predictor = EnhancedTravelRecommendationPredictor(model_path)
    recommendations = predictor.get_recommendations(min_cost, max_cost, travel_month, travel_place)
    
    if recommendations:
        predictor.display_recommendations(recommendations)
    
    return recommendations

def get_available_options(model_path='main_model.pkl'):
    """Get available travel options from the enhanced trained model"""
    predictor = EnhancedTravelRecommendationPredictor(model_path)
    return predictor.get_available_options()

def get_satisfaction_stats(travel_place=None, model_path='main_model.pkl'):
    """Get satisfaction statistics"""
    predictor = EnhancedTravelRecommendationPredictor(model_path)
    predictor.get_satisfaction_stats(travel_place)

# Enhanced interactive mode
def interactive_mode():


    print("ENHANCED TRAVEL RECOMMENDATION SYSTEM - INTERACTIVE MODE")
    print("WITH USER SATISFACTION DATA")
    
    predictor = EnhancedTravelRecommendationPredictor()
    
    if predictor.hotel_model is None:
        print("Enhanced model could not be loaded. Exiting...")
        return
    


    try:
        print("\n" + "-"*70)
        print("Enter your travel preferences:")
            
        # Get user input
        user_input = input("Travel month: ").strip()
                
        travel_month = user_input
        travel_place = input("Travel place: ").strip()
                
        min_cost = int(input("Minimum budget per person (BDT): "))
        max_cost = int(input("Maximum budget per person (BDT): "))
            
            # Make enhanced prediction
        recommendations = predictor.get_recommendations(min_cost, max_cost, travel_month, travel_place)
        predictor.display_recommendations(recommendations)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except ValueError:
        print("Please enter valid numbers for budget.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":

     interactive_mode()