import json

def display_json_example():
    """
    A simple example showing the JSON structure used in prompt chaining trends.
    """
    trends_data = {
        "trends": [
            {
                "trend_name": "AI-Powered Personalization",
                "supporting_data": "73% of consumers prefer to do business with brands that use personal information to make their shopping experiences more relevant."
            },
            {
                "trend_name": "Sustainable and Ethical Brands",
                "supporting_data": "Sales of products with ESG-related claims grew 28% over the last five years, compared to 20% for products without."
            }
        ]
    }
    
    print("--- Trends JSON Example ---")
    print(json.dumps(trends_data, indent=2))

if __name__ == "__main__":
    display_json_example()
