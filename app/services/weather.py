def check_weather(location: str, days: int = 1) -> dict:
    """
    Mock weather tool that returns a standardized weather report.
    In a real app, this would call an external API like OpenWeatherMap.
    """
    # Simply mocking response for demonstration
    weather_data = {
        "location": location,
        "forecast": []
    }
    
    # Mock data generation
    base_temp = 20
    conditions = ["Sunny", "Cloudy", "Rain", "Windy"]
    
    for i in range(days):
        day_report = {
            "day": i + 1,
            "condition": conditions[i % len(conditions)],
            "temperature_c": base_temp + (i % 5),
            "humidity": 50 + (i * 2),
            "wind_speed_kmh": 10 + (i * 3)
        }
        weather_data["forecast"].append(day_report)
        
    return weather_data
