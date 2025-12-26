import pandas as pd
import numpy as np
import joblib
import sys


def predict_car_price(model, scaler, freq_maps, model_columns, 
                      year, mileage, engine, make, model_name, registered, 
                      fuel, transmission, assembly, body, color):
    
    # Create a DataFrame for the single new car
    new_data = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'engine': [engine],
        'make': [make.lower()],
        'model': [model_name.lower()],
        'registered': [registered.lower()],
        'fuel': [fuel.lower()],
        'transmission': [transmission.lower()],
        'assembly': [assembly.lower()],
        'body': [body.lower()],
        'color': [color.lower()]
    })

    # Apply Frequency Encoding (High Cardinality)
    for col in ['make', 'model', 'registered']:
        val = new_data[col].iloc[0]
        # Look up the frequency score from the saved map. If unseen, use 0.001
        if val in freq_maps[col].index:
            new_data[col] = freq_maps[col][val]
        else:
            new_data[col] = 0.001 

    # Apply One-Hot Encoding (Low Cardinality)
    # Re-apply the same get_dummies process used during training
    new_data = pd.get_dummies(new_data, columns=['fuel', 'transmission', 'assembly', 'body', 'color'], drop_first=True)

    # Reindex to match the training column structure 
    # This ensures the input data has the EXACT same columns as the model was trained on,
    # filling any missing dummy columns (like 'fuel_Diesel') with 0.
    new_data = new_data.reindex(columns=model_columns, fill_value=0)

    # Scale the Data
    new_data_scaled = scaler.transform(new_data)

    # Predict using the trained Random Forest model
    predicted_price = model.predict(new_data_scaled)[0]
    
    return predicted_price


def get_user_input():
    
    print("\n--- Enter Car Specifications for Prediction ---")
    
    try:
        year = int(input("1. Manufacturing Year (e.g., 2022): "))
        mileage = int(input("2. Mileage (km, e.g., 50000): "))
        engine = int(input("3. Engine Capacity (cc, e.g., 1800): "))
    except ValueError:
        print("Invalid input for Year, Mileage, or Engine. Please enter integers.")
        sys.exit(1)
        
    make = input("4. Make (e.g., Honda): ")
    model_name = input("5. Model (e.g., Civic): ")
    registered = input("6. Registration City/Status (e.g., Islamabad/unregistered): ")
    fuel = input("7. Fuel Type (e.g., Petrol/Diesel/Electric): ")
    transmission = input("8. Transmission (Automatic/Manual): ")
    assembly = input("9. Assembly (Local/Imported): ")
    body = input("10. Body Type (e.g., Sedan/Hatchback): ")
    color = input("11. Color (e.g., White/Black): ")
    
    return (year, mileage, engine, make, model_name, registered, fuel, transmission, assembly, body, color)


def main():
    
    model_path = 'final_car_price_model.pkl' 
    
    try:
        package = joblib.load(model_path)
        rf_model = package['model']
        scaler = package['scaler']
        freq_maps = package['freq_maps']
        model_columns = package['model_columns']
        
    except FileNotFoundError:
        print(f"\nError: Model file '{model_path}' not found.")
        print("Please ensure you have saved your final model in the notebook.")
        return

    # Get user input
    try:
        (year, mileage, engine, make, model_name, registered, fuel, transmission, assembly, body, color) = get_user_input()
    except SystemExit:
        return

    # Run the prediction
    try:
        price = predict_car_price(rf_model, scaler, freq_maps, model_columns,
                                  year, mileage, engine, make, model_name, registered, 
                                  fuel, transmission, assembly, body, color)
        
        # Print result
        print("\n=======================================================")
        print("Your Predicted Car Price:")
        print(f"   {price:,.0f} PKR")
        print("=======================================================")
        
    except Exception as e:
        print(f"\nPrediction failed due to an internal error: {e}")
        print("Ensure all input features are valid and match expected types.")


if __name__ == "__main__":
    main()