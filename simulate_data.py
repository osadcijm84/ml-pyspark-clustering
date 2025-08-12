import os
import json

def simulate_data_loading(output_path="simulated_openfoodfacts_data.jsonl", num_records=1000):
    """Simulates loading and preprocessing a small Open Food Facts dataset."""
    print(f"Simulating data loading and preprocessing for {num_records} records...")
    
    # Create a dummy JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_records):
            record = {
                "_id": f"product_{i}",
                "product_name": f"Simulated Product {i}",
                "nutriscore_score": i % 5, # Example numerical feature
                "energy_100g": float(i * 10),
                "proteins_100g": float(i % 20),
                "fat_100g": float(i % 30),
                "carbohydrates_100g": float(i % 40),
                "ingredients_text": f"ingredient A, ingredient B, ingredient C for product {i}"
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"Simulated data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    simulate_data_loading()


