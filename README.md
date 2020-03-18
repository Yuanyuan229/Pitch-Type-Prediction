# Pitch-Type-Prediction

Sports Analytics - Pitch Type Prediction for Baseball

If a hitter can gain even a slight improvement in guessing which pitch is coming next, his success could increase dramatically. A useful prediction model could aid teams in tipping their hitters on what to expect. Conversely, it could be utilized for identifying where their own pitchers may be predictable.

## data source
MLB Pitch Data 2015-2018 https://www.kaggle.com/pschale/mlb-pitch-data-20152018
2018 MLB Standard Pitching https://www.baseball-reference.com/leagues/MLB/2018-standard-pitching.shtml
Pitcher Info Sample Page https://www.baseball-reference.com/players/b/bauertr01.shtml

## web scrapping
Refer to file: 
Pitch Data - Web Scrapping and Database creation.ipynb

Besides using the cleaned pitch data from Kaggle, we also did web scrapping to get more features and make the data more comprehensive.

## Prediction model
Refer to file:
Pitch Type Prediction.ipynb

Since the ability to predict pitch provides such a significant advantage to batters, in this project, we ran several classification models controling for differences in leagues and teams. 

In order to predict pitch type, individual pitch data from all major league games from 2015-2018 are utilised. This data included over 2.8 million rows, each representing an individual pitch. Each pitch was self categorized into one of twenty-two different possible pitch type labels.  

Extremely uncommon pitches like Eephus pitches are filtered out, and then the remaining pitch types are categorized into three broader categories: fastballs, breaking balls, and change ups.  After this adjustment, the data become more balanced and is ready use.

## Conclusion

Inspired by the Astros’ sign-stealing scandal, this project uses machine learning techniques to predict pitch type. This analysis uses a random forest model to predict pitches in multiple different scenarios. From these different scenarios it is recommended that on average batters should consider whether or not a runner is on third to receive a fastball and right handed pitchers pitching towards a left handed hitter with low ball count is most likely to throw a fastcall regardless of pitch number or a running being on base. In some of our more filtered models, like in our all pitchers versus Astros batters model (Appendix C), we can say having a high ball count with a runner on third means that an Astros batter is most likely to see a fastball hurled over home plate (assuming they are a right handed batter). 

For future analysis, pitchers can be divided into different positions such as starter, relief, and closer. These different pitching positions vary in the average number of pitches per game and therefore some positions may favor different pitch types based on different game contexts. Additionally, coaches may want to break down this model to a pitcher specific level for pitchers that their team face regularly, potentially pitchers from other teams in their own divisions. This will allow batters to have a better fit edge on a higher number at bats they face opposed to a generalized recommendation. Lastly, machine learning models for pitch prediction need evolve as more data becomes available. 
