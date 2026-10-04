

def risk_level(score: float) -> str:
    if score<0.3 :
        return "low"
    if score<0.7:
        return "medium"
    return "high"

def recommendation(level: str) -> str:
    if level == "low":
        return "Low risk level - safe option"
    if level == "medium":
        return "Medium risk level - please be advised when making a descision"
    return "High risk level - it is highly recommended to postpone this deal untill the risk level drops significantly"