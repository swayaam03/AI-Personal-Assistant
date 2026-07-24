import random
from langchain_core.tools import tool

# Mock weather database for popular locations
_MOCK_WEATHER = {
    "london": {"condition": "Light Rain & Cloudy", "temp": "16°C", "humidity": "78%"},
    "new york": {"condition": "Sunny", "temp": "24°C", "humidity": "45%"},
    "tokyo": {"condition": "Clear Sky", "temp": "22°C", "humidity": "50%"},
    "paris": {"condition": "Partly Cloudy", "temp": "19°C", "humidity": "60%"},
    "mumbai": {"condition": "Humid & Sunny", "temp": "31°C", "humidity": "82%"},
    "san francisco": {"condition": "Foggy", "temp": "17°C", "humidity": "75%"},
    "delhi": {"condition": "Warm", "temp": "34°C", "humidity": "40%"},
}


@tool
def get_weather(location: str) -> str:
    """
    Fetches the current weather report for a given city or location.
    
    Args:
        location: The city or region name, e.g. "London", "Tokyo", "New York".
        
    Returns:
        A weather report summary including temperature, conditions, and humidity.
    """
    clean_location = location.strip().lower()
    
    # Check mock database
    if clean_location in _MOCK_WEATHER:
        data = _MOCK_WEATHER[clean_location]
        return (
            f"Weather Report for {location.title()}:\n"
            f"- Temperature: {data['temp']}\n"
            f"- Condition: {data['condition']}\n"
            f"- Humidity: {data['humidity']}"
        )
    
    # Fallback default generator for un-mocked cities
    temp = random.randint(18, 28)
    humidity = random.randint(45, 70)
    conditions = ["Sunny", "Partly Cloudy", "Mild Breezes", "Overcast"]
    selected_condition = random.choice(conditions)
    
    return (
        f"Weather Report for {location.title()}:\n"
        f"- Temperature: {temp}°C\n"
        f"- Condition: {selected_condition}\n"
        f"- Humidity: {humidity}%"
    )
