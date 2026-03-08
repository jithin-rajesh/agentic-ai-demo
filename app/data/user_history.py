"""
Dummy user profiles and ride history for the Sensorless Cycling Coach.
In production this would come from a database; here we hard-code three
representative riders so every pipeline node has realistic context.
"""

USERS = {
    # ── Rider 1: Enthusiastic beginner ───────────────────────────────
    "rider_001": {
        "name": "Alex",
        "age": 28,
        "weight_kg": 75,
        "ftp_watts": 180,          # Functional Threshold Power
        "experience_level": "beginner",
        "location": "Bangalore",
        "bike_type": "road",
        "preferences": {
            "terrain": "flat",
            "max_ride_hours": 2.0,
            "preferred_days": ["Tuesday", "Thursday", "Saturday", "Sunday"],
            "goal": "Build endurance and complete first century ride",
        },
        "ride_history": [
            {"date": "2026-02-24", "distance_km": 25, "duration_min": 70,
             "avg_power_w": 130, "avg_hr": 145, "elevation_m": 120,
             "route": "Cubbon Park Loop", "notes": "Felt good, easy pace"},
            {"date": "2026-02-22", "distance_km": 18, "duration_min": 55,
             "avg_power_w": 125, "avg_hr": 140, "elevation_m": 80,
             "route": "Ulsoor Lake Circuit", "notes": "Recovery ride"},
            {"date": "2026-02-20", "distance_km": 35, "duration_min": 100,
             "avg_power_w": 145, "avg_hr": 155, "elevation_m": 200,
             "route": "Nandi Hills Approach", "notes": "First hilly ride, legs tired"},
            {"date": "2026-02-17", "distance_km": 22, "duration_min": 65,
             "avg_power_w": 128, "avg_hr": 142, "elevation_m": 90,
             "route": "Hebbal Lake Loop", "notes": "Steady effort"},
            {"date": "2026-02-15", "distance_km": 30, "duration_min": 90,
             "avg_power_w": 140, "avg_hr": 150, "elevation_m": 150,
             "route": "Airport Road Out-and-Back", "notes": "Pushed harder today"},
            {"date": "2026-02-12", "distance_km": 15, "duration_min": 45,
             "avg_power_w": 120, "avg_hr": 135, "elevation_m": 50,
             "route": "Easy Spin", "notes": "Very light session"},
        ],
        "health_notes": "No injuries. Mild knee discomfort after long rides.",
        "weekly_volume_target_km": 80,
    },

    # ── Rider 2: Intermediate club racer ─────────────────────────────
    "rider_002": {
        "name": "Priya",
        "age": 34,
        "weight_kg": 62,
        "ftp_watts": 240,
        "experience_level": "intermediate",
        "location": "Hyderabad",
        "bike_type": "road",
        "preferences": {
            "terrain": "mixed",
            "max_ride_hours": 3.5,
            "preferred_days": ["Monday", "Wednesday", "Friday", "Saturday", "Sunday"],
            "goal": "Improve FTP and race in local criterium series",
        },
        "ride_history": [
            {"date": "2026-02-25", "distance_km": 72, "duration_min": 150,
             "avg_power_w": 195, "avg_hr": 155, "elevation_m": 650,
             "route": "Shamirpet Long Loop", "notes": "Good endurance ride with club"},
            {"date": "2026-02-23", "distance_km": 40, "duration_min": 80,
             "avg_power_w": 220, "avg_hr": 165, "elevation_m": 300,
             "route": "Tempo Intervals", "notes": "4x10min at threshold"},
            {"date": "2026-02-21", "distance_km": 25, "duration_min": 50,
             "avg_power_w": 160, "avg_hr": 130, "elevation_m": 100,
             "route": "Recovery Spin", "notes": "Easy legs"},
            {"date": "2026-02-19", "distance_km": 55, "duration_min": 120,
             "avg_power_w": 200, "avg_hr": 158, "elevation_m": 500,
             "route": "KBR Park + Climb", "notes": "Hill repeats felt strong"},
            {"date": "2026-02-17", "distance_km": 85, "duration_min": 195,
             "avg_power_w": 185, "avg_hr": 148, "elevation_m": 800,
             "route": "Weekend Century Prep", "notes": "Longest ride this month"},
            {"date": "2026-02-15", "distance_km": 35, "duration_min": 70,
             "avg_power_w": 230, "avg_hr": 170, "elevation_m": 250,
             "route": "VO2max Intervals", "notes": "5x3min hard efforts"},
            {"date": "2026-02-13", "distance_km": 20, "duration_min": 40,
             "avg_power_w": 155, "avg_hr": 125, "elevation_m": 80,
             "route": "Active Recovery", "notes": "Legs still heavy from intervals"},
            {"date": "2026-02-11", "distance_km": 60, "duration_min": 130,
             "avg_power_w": 190, "avg_hr": 152, "elevation_m": 450,
             "route": "Sweet-spot Ride", "notes": "2x20min SST, felt manageable"},
        ],
        "health_notes": "Previous hamstring strain (healed). Watch left knee on steep climbs.",
        "weekly_volume_target_km": 200,
    },

    # ── Rider 3: Veteran long-distance tourer ────────────────────────
    "rider_003": {
        "name": "Ravi",
        "age": 45,
        "weight_kg": 82,
        "ftp_watts": 210,
        "experience_level": "advanced",
        "location": "Chennai",
        "bike_type": "touring",
        "preferences": {
            "terrain": "hilly",
            "max_ride_hours": 5.0,
            "preferred_days": ["Wednesday", "Saturday", "Sunday"],
            "goal": "Complete a multi-day 500km coastal tour",
        },
        "ride_history": [
            {"date": "2026-02-26", "distance_km": 110, "duration_min": 270,
             "avg_power_w": 170, "avg_hr": 140, "elevation_m": 950,
             "route": "ECR to Mahabalipuram", "notes": "Headwind on return, solid effort"},
            {"date": "2026-02-23", "distance_km": 45, "duration_min": 100,
             "avg_power_w": 155, "avg_hr": 130, "elevation_m": 200,
             "route": "Besant Nagar Loop", "notes": "Easy with touring load"},
            {"date": "2026-02-22", "distance_km": 80, "duration_min": 200,
             "avg_power_w": 165, "avg_hr": 138, "elevation_m": 600,
             "route": "Yelagiri Climb", "notes": "Tested loaded panniers on climbs"},
            {"date": "2026-02-19", "distance_km": 30, "duration_min": 75,
             "avg_power_w": 140, "avg_hr": 125, "elevation_m": 100,
             "route": "City Recovery", "notes": "Bike maintenance after"},
            {"date": "2026-02-16", "distance_km": 130, "duration_min": 330,
             "avg_power_w": 160, "avg_hr": 135, "elevation_m": 1100,
             "route": "Coastal Century", "notes": "Practiced nutrition strategy"},
            {"date": "2026-02-14", "distance_km": 50, "duration_min": 120,
             "avg_power_w": 150, "avg_hr": 128, "elevation_m": 350,
             "route": "Tempo w/ Touring Weight", "notes": "Good simulation ride"},
            {"date": "2026-02-12", "distance_km": 25, "duration_min": 60,
             "avg_power_w": 135, "avg_hr": 120, "elevation_m": 80,
             "route": "Easy Spin", "notes": "Recovery day"},
        ],
        "health_notes": "Managed lower back pain — needs off-bike stretching. Strong cardiovascular base.",
        "weekly_volume_target_km": 250,
    },
}


def get_user_profile(user_id: str) -> dict:
    """Return a user profile by ID or the default rider."""
    return USERS.get(user_id, USERS["rider_001"])


def get_all_user_ids() -> list[str]:
    """Return all available user IDs."""
    return list(USERS.keys())


def get_user_summary(user_id: str) -> str:
    """Return a short plain-text summary of the user for LLM context."""
    u = get_user_profile(user_id)
    recent = u["ride_history"][:5]
    total_km = sum(r["distance_km"] for r in recent)
    avg_power = round(sum(r["avg_power_w"] for r in recent) / len(recent)) if recent else 0

    return (
        f"Rider: {u['name']}, Age: {u['age']}, Weight: {u['weight_kg']}kg, "
        f"FTP: {u['ftp_watts']}W, Level: {u['experience_level']}, "
        f"Location: {u['location']}, Bike: {u['bike_type']}\n"
        f"Goal: {u['preferences']['goal']}\n"
        f"Preferred days: {', '.join(u['preferences']['preferred_days'])}\n"
        f"Max ride duration: {u['preferences']['max_ride_hours']}h, "
        f"Terrain preference: {u['preferences']['terrain']}\n"
        f"Weekly target: {u['weekly_volume_target_km']}km\n"
        f"Last 5 rides: {total_km}km total, {avg_power}W avg power\n"
        f"Health: {u['health_notes']}"
    )
