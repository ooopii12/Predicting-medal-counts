# Predicting-medal-counts
#Predicting on Olympics game
Project Title: Predicting Olympic Medal Counts: A Comparative Analysis of Statistical and Machine Learning Models

# Objective:
The primary goal of this project is to develop predictive models that estimate the number of medals won by each country at the Rio Olympics in 2016 using data from 2012 Olympic Games and other related socio-economic variables. This involves identifying key factors that influence Olympic performance, evaluating the effectiveness of different statistical models, and suggesting ways to improve predictions for future Olympic events.

# Background:
The Olympic Games are a global event where nations compete across various sports. Predicting the number of medals a country might win is a complex task, influenced by numerous factors such as economic strength, population, historical performance, and political background. Accurate predictions can be valuable for national sports committees, sponsors, and analysts interested in understanding the determinants of Olympic success.

In this project, we use data from previous Olympics, particularly focusing on the 2012 London Olympics, to build models that predict the medal counts for the 2016 Rio Olympics. By comparing different modeling approaches, including Poisson regression, Negative Binomial regression, and Gradient Boosting Machines (GBM), we aim to determine which methods offer the most accurate predictions and why.

# Data available
Data are available on the number of medals (total and gold) won by each country for 108 coun tries participating in the Rio 2016 Olympics, along with information on previous Olympic performance (from the 2000, 2004, 2008 and 2012 Games) and other variables.
It is also possible to augment the data by adding variables to the list below, provided that these variables were available before the beginning of the Games in August 2016.
The dataset olympics2016. csv has 108 observations and the following variables:

• country the country's name

• country. code the country's three-letter code

• gapYY the country's GPD in millions of US dollars during year VY

• popYY the country's population in thousands in year YY

• soviet 1 if the country was part of the former Soviet Union, 0 otherwise

• comm 1 if the country is a former/current communist state, 0 otherwise

• muslim 1 if the country is a Muslim majority country, 0 otherwise

• oneparty 1 if the country is a one-party state, 0 otherwise

• goldYY number of gold medals won in the YY Olympics

• totYY total number of medals won in the YY Olympics

• totgoldYY overall total number of gold medals awarded in the YY Olympics

• totmedalsYY overall total number of all medals awarded in the YY Olympics

• bmi average BMI (not differentiating by gender),

• altitude altitude of the country's capital city,

• athletesYY number of athletes representing the country in the YY Olympics,

• host 1 if the country has hosted/is hosting/will be hosting the Olympics, 0 otherwise.

# question
• Which variables are associated with the number of medals (total) won in the 2012 Olympics?

• How well does a model based on data up to and including 2012 predict Olympic performance in the 2016 Games?

• What improvements might be made to the model/data collected in order to better predict Olympic medal counts for future games?
