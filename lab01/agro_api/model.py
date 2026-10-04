from schemas import FarmRequest

def calculate_risk(data: FarmRequest) -> float:
    score = 0.1 
    if data.payment_delay_day > 30:
        score += 0.3
    if data.previous_defaults > 0:
        score += 0.3
    if data.debt > 5_000_000:
        score += 0.2
    if data.precipitation_mm < 100:
        score += 0.1
    return min(score, 1.0)


