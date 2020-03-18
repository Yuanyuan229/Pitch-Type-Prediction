# Pitch-Type-Prediction

Sports Analytics - Pitch Type Prediction for Baseball

If a hitter can gain even a slight improvement in guessing which pitch is coming next, his success could increase dramatically. A useful prediction model could aid teams in tipping their hitters on what to expect. Conversely, it could be utilized for identifying where their own pitchers may be predictable.


## data source
Since the ability to predict pitch provides such a significant advantage to batters, in this project, we ran several classification models controling for differences in leagues and teams. 

In order to predict pitch type, individual pitch data from all major league games from 2015-2018 are utilised. This data included over 2.8 million rows, each representing an individual pitch. Each pitch was self categorized into one of twenty-two different possible pitch type labels.  

Extremely uncommon pitches like Eephus pitches are filtered out, and then the remaining pitch types are categorized into three broader categories: fastballs, breaking balls, and change ups.  After this adjustment, the data become more balanced and is ready use.
