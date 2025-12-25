AFCON 2025 Winner Prediction Model
Project Overview
A comprehensive machine learning system for predicting match outcomes and tournament results for the Africa Cup of Nations (AFCON) 2025. 
The model employs ensemble techniques combining multiple predictive approaches to forecast match scores, group stage standings, knockout stage progression, and ultimately the tournament champion.

Mathematical Framework and Prediction Methodology
**1. Data Integration and Feature Engineering**

Team-Level Features:
FIFA Ranking: International ranking points (inverse scaling applied)
Attack Strength: Composite metric derived from historical goal scoring
Defense Strength: Composite metric derived from historical goals conceded
Home Advantage: Binary indicator with weighted coefficient for host nation
CAF Ranking: Confederation-specific performance ranking

Player-Level Features:
Average Player Rating: Mean rating across squad players
Maximum Player Rating: Highest individual player rating
Rating Standard Deviation: Measure of squad consistency
Total Goals/Assists: Aggregate offensive contributions
Average Squad Age: Team age demographic analysis
Squad Size: Number of registered players

Form-Based Features:
Form Points Calculation: Form Points = Σ(char ∈ form_string) { 'W': 3, 'D': 1, 'L': 0 }
Goals Scored/Conceded (Last 5): Rolling offensive/defensive performance
Goal Difference Trend: Directional momentum indicator

Head-to-Historical Features:
Historical Win Rate: Win Rate_team1 = wins_team1 / total_matches
Average Goals Scored: Historical offensive performance against specific opponent
Draw Probability: Draw Rate = draws / total_matches


**2. Expected Goals (xG) Model**
   
Base xG Calculation: xG_team1 = (attack_strength_team1 / 100) × league_avg_goals × (1 - (defense_strength_team2 / 100) × 0.5)  Where league_avg_goals = 1.4 (derived from historical AFCON data)
Form-Adjusted xG: xG_adj = base_xG × [1 + (form_points / max_form_points) × 0.2]
Historical Adjustment: final_xG = (current_xG + historical_avg_goals) / 2


**3. Score Prediction Models**

   
  3.1 Random Forest Regression Models
Model 1: Predicts Team 1 score

Model 2: Predicts Team 2 score
Parameters:
Number of estimators: 100-200
Maximum depth: 8-10
Minimum samples split: 5
Minimum samples leaf: 2
Random state: 42

  3.2 Poisson Distribution Adjustment
Football scores follow a Poisson distribution: P(k goals) = (λ^k × e^{-λ}) / k! Where λ = expected goals (xG)
Implementation:
def poisson_adjustment(prediction):
    lambda_param = max(0.1, prediction)
    poisson_probs = [np.exp(-lambda_param) * (lambda_param**k) / factorial(k) for k in range(6)]
    cum_probs = np.cumsum(poisson_probs)
    rand_val = np.random.random()
    for goals, cum_prob in enumerate(cum_probs):
        if rand_val <= cum_prob:
            return goals
    return min(5, int(round(prediction)))


**4. Ensemble Model Architecture**

   
Weighted Ensemble Approach: Final Score = (Original Model × 0.6) + (Enhanced Model × 0.4)
Weight allocation based on model performance:
Original Model: 91.7% winner accuracy (higher weight for outcome prediction)
Enhanced Model: Better score distribution realism (lower weight for calibration)

Decision Logic:
1/ If both models predict same winner → Weighted average
2/ If enhanced predicts draw but original predicts clear winner (diff ≥ 2) → Trust original
3/ If original predicts draw but enhanced predicts clear winner → Trust enhanced
4/ Special handling for 0-0 predictions with significant strength differentials
    


**5. Knockout Stage Simulation**


Extra Time Simulation:
Applied when predicted score is tied after 90 minutes
Extra time factor: 0.7 (scoring probability reduction)
Extra time goal probability: 30%
Goal allocation weighted by team attack strength

Penalty Shootout Simulation:
Penalty win probability calculated as weighted average: P(penalty_win) = (attack_factor × 0.4) + (rating_factor × 0.3) + (form_factor × 0.3)
Where:
attack_factor = team1_attack / (team1_attack + team2_attack)
rating_factor = team1_avg_rating / (team1_avg_rating + team2_avg_rating)
form_factor = team1_form_pts / (team1_form_pts + team2_form_pts)


**6. Tournament Bracket Generation**

   
Group Stage Qualification:
1/ Top 2 teams from each group qualify automatically
2/ Best 4 third-place teams qualify based on:
Points
Goal difference
Goals scored
Goals conceded

Knockout Stage Pairing:
Following AFCON 2023 format:
1. 1A vs 3C/D/E
2. 2D vs 2E
3. 1B vs 3A/D/E/F
4. 1F vs 3A/B/C
5. 1C vs 3A/B/F
6. 1E vs 2D
7. 1D vs 3B/E/F
8. 2A vs 2B


**7. Model Performance Metrics**
Evaluation on Completed Matches:
Winner Prediction Accuracy: 91.7% (11/12 correct)
Exact Score Accuracy: 25.0% (3/12 exact)
Average Score Difference Error: 0.33 goals

Knockout Stage Statistics:
Total matches simulated: 14
Regular time decisions: 12 matches (85.7%)
Extra time decisions: 1 match (7.1%)
Penalty decisions: 1 match (7.1%)
Average goals per match: 2.71




**8. Final Tournament Results**
Champion: Morocco
Runner-up: Algeria
Final Score: Morocco 2-1 Algeria

Complete Standings:
Morocco - Champion
Algeria - Runner-up

Ivory Coast - Semifinalist
Senegal - Semifinalist

5-8. Quarterfinalists: DR Congo, Egypt, Mali, Zambia
9-16. Round of 16 Participants: Burkina Faso, DR Congo, Equatorial Guinea, Mali, Nigeria, South Africa, Uganda




**9. Technical Implementation Details**

Data Processing Pipeline:
1/Data validation and cleaning
2/Feature extraction and normalization
3/Missing value imputation using mean substitution
4/Categorical variable encoding


Model Training:
Training/Test split: 80/20
Cross-validation: 5-fold
Feature scaling: StandardScaler applied to continuous variables
Hyperparameter tuning: Grid search on Random Forest parameters


Prediction Pipeline:
1/ Feature vector construction for match pair
2/ Individual model predictions
3/ Ensemble combination
4/ Poisson distribution adjustment
5/ Final score rounding and validation



**10. Model Limitations and Future Improvements**


Current Limitations:
Limited historical data for some team pairings
Player form data based on club performance, not international
Injury data and squad availability not incorporated
Weather conditions and venue effects simplified

Potential Improvements:
Incorporate advanced metrics: xG chain, expected assists (xA)
Add player fatigue metrics from club minutes played
Include manager tactics and formation data
Implement time-series analysis for team form
Add betting market odds as predictive feature




**11. Deployment and Usage**

Requirements:
Python 3.8+
scikit-learn 1.0+
pandas 1.3+
numpy 1.21+

Updating Predictions:
To update predictions with new match results:
Add completed match data to data/completed_matches.csv
Update team form in data/team_current_form.csv
Run the prediction pipeline to regenerate forecasts


**12. Ethical Considerations**
Predictions for Informational Purposes Only: Model outputs should not be used for gambling or betting
Data Privacy: All player data sourced from publicly available statistics
Bias Mitigation: Regular model auditing for systematic prediction biases
Transparency: Full disclosure of methodology and assumptions


**13. Citation**
If using this model for research or publication, please cite:
AFCON 2025 Prediction Model. Ensemble machine learning system for football tournament forecasting.
Mathematical framework incorporating xG modeling, Poisson distributions, and weighted ensemble techniques.


**15. Contact**
For questions, issues, or collaboration requests:
Email: malek.dhaouadi@esprit.tn
Linkedin: https://www.linkedin.com/in/malek-dhaouadi-74a747246/


-----------------------------------------------------------------------------------------------------------------------------------------------------------

Last Updated: 25 December 2025
Model Version: 1.0
Prediction Accuracy: 91.7% winner prediction rate
