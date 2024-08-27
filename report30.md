Nian He
30/08/2024

## Predicting the 2016 Olympic Medals Using General Linear Models and Ensemble Machine Learning Techniques

------------------------------------------------------------------------

## 1. Introduction

The Olympic Games serve as a global platform where nations showcase
their athletic prowess, with the total number of medals won by each
country often reflecting its sports development, resources, and
strategic investments. Predicting the total number of medals a country
will win at an upcoming Olympic event is a complex task that requires
careful consideration of various factors, including economic indicators,
historical performance, and geopolitical context.

This project aimed to develop predictive models to forecast the total
number of medals that countries won at the Rio 2016 Olympics using data
available before the Games. The focus was on identifying key factors
that influence Olympic success and evaluating the accuracy of the
predictive models. The ultimate goal was to provide insights that can
improve future predictions of Olympic medal counts.

The questions of interest are the following:

- Which variables are associated with the total number of medals won by
  each country in 2012?
- How well does the model predict the 2016 results?
- What improvements might be made to the model/data collected to
  increase the predicting performance?

The specific objectives of this project are:

- Identify Key Variables: Analyzed the variables associated with the
  total number of medals won in the 2012 Olympics to determine the
  factors influencing medal counts.

- Develop Predictive Models: Built and evaluate models using data up to
  and including the 2012 Olympics to predict the total medal counts for
  the 2016 Rio Olympics.

- Assess Model Performance: Compared the model predictions with the
  actual outcomes of the 2016 Olympics and with predictions published
  online before the Games.

- Suggest Improvements: Proposed enhancements to the models and data
  collection for better prediction accuracy in future Olympics.

## 2.Data description

The dataset `olympics2016.csv` includes data on countries that
participated in the 2016 Rio Olympics, with the following variables (all
YY variables correspond to the years 2000, 2004, 2008, 2012, and 2016):

### Numerical Variables:

- **gdpYY**: The country’s GDP in millions of US dollars for the year
  YY.
- **popYY**: The country’s population in thousands for the year YY.
- **goldYY**: Number of gold medals won by the country in the YY
  Olympics.
- **totYY**: Total number of medals won by the country in the YY
  Olympics.
- **totgoldYY**: Total number of gold medals awarded in the YY Olympics.
- **totmedalsYY**: Total number of all medals awarded in the YY
  Olympics.
- **bmi**: Average BMI (Body Mass Index) of the population (not
  distinguishing by gender).
- **altitude**: Altitude of the country’s capital city above sea level.
- **athletesYY**: Number of athletes representing the country in the YY
  Olympics.

### Categorical Variables:

- **country**: Name of the country.
- **country.code**: The three-letter code representing the country.
- **soviet**: Indicates whether the country was part of the former
  Soviet Union (1 = Yes, 0 = No).
- **comm**: Indicates whether the country is a former or current
  communist state (1 = Yes, 0 = No).
- **muslim**: Indicates whether the country has a Muslim-majority
  population (1 = Yes, 0 = No).
- **oneparty**: Indicates whether the country is a one-party state (1 =
  Yes, 0 = No).
- **host**: Indicates whether the country has hosted, is hosting, or
  will host the Olympics (1 = Yes, 0 = No).

## 3. Methodology

This project employed a combination of statistical and machine learning
methodologies to predict the total number of medals won by each country
in the 2016 Rio Olympics.

### 3.1 Exploratory Data Analysis (EDA)

**Objective:**  
Before constructing predictive models, we first conducted Exploratory
Data Analysis (EDA) to understand the relationships between variables,
identified significant predictors, and assessed the distributions of key
variables.

**Key Methods:**

- **Correlation Analysis:** Correlation analysis was used to identify
  linear relationships between continuous variables, such as GDP,
  population, and the number of medals won. The correlation coefficient
  ($r$) quantified the strength of these relationships:

$$
  r = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2 \sum_{i=1}^{n} (Y_i - \bar{Y})^2}}
  $$

- **Boxplots and Chi-Squared Tests for Categorical Variables:**  
  Boxplots were used to visualize the distribution of medals across
  different categorical variables (e.g., former Soviet Union status,
  hosting status). Chi-squared tests assessed the independence between
  categorical variables:

  $$
  \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
  $$

  where $O_i$ is the observed frequency, and $E_i$ was the expected
  frequency under the null hypothesis of independence.

### 3.2 General Linear Models (GLM)

**Objective:**  
To identify the key variables associated with the number of medals won
and to develop a predictive model based on data up to and including the
2012 Olympics.

**Key Methods:**

- **Poisson Regression:**  
  Given that the number of medals was a count variable, Poisson
  regression was initially employed. The Poisson model assumed that the
  count data follows a Poisson distribution:

$$
  P(Y = y) = \frac{e^{-\lambda} \lambda^y}{y!}
  $$

where $\lambda = e^{\beta_0 + \beta_1X_1 + \dots + \beta_kX_k}$ was the
mean and variance of the distribution, with $X_1, \dots, X_k$ being the
predictors.

- **Negative Binomial Regression:**  
  To account for overdispersion in the count data (variance greater than
  the mean), a Negative Binomial model was also applied. This model
  generalized the Poisson regression by adding a dispersion parameter
  $\alpha$, allowing variance to exceed the mean:

  $$
  \text{Var}(Y) = \lambda + \alpha \lambda^2
  $$

### 3.3 Zero-Inflated Models

**Objective:**  
Zero-Inflated models were particularly useful when dealing with count
data that have an excess number of zeros. These models assume that the
data come from two different processes: one that generates only zeros
(e.g., countries that do not participate in medal-winning sports) and
one that generates counts following a Poisson or Negative Binomial
distribution (e.g., countries that do participate).

**Key Methods:**

- **Zero-Inflated Poisson Model (ZIP):**  
  The Zero-Inflated Poisson model combined a Poisson distribution with a
  logit model that predicts the probability of an excess zero:

$$
  P(Y = 0) = \pi + (1 - \pi) \cdot e^{-\lambda}
  $$

$$
  P(Y = y) = (1 - \pi) \cdot \frac{e^{-\lambda} \lambda^y}{y!}, \quad y > 0
  $$

where $\pi$ was the probability of the zero-inflation component and
$\lambda$ was the mean of the Poisson distribution.

- **Zero-Inflated Negative Binomial Model (ZINB):**  
  The Zero-Inflated Negative Binomial model was similar to the ZIP model
  but used a Negative Binomial distribution to account for
  overdispersion:

  $$
  P(Y = 0) = \pi + (1 - \pi) \cdot \left(\frac{1}{1 + \alpha \lambda}\right)^{\frac{1}{\alpha}}
  $$

  $$
  P(Y = y) = (1 - \pi) \cdot \frac{\Gamma(y + \frac{1}{\alpha})}{\Gamma(\frac{1}{\alpha}) y!} \left(\frac{1}{1 + \alpha \lambda}\right)^{\frac{1}{\alpha}} \left(\frac{\alpha \lambda}{1 + \alpha \lambda}\right)^y, \quad y > 0
  $$

  where $\alpha$ was the dispersion parameter, and $\lambda$ was the
  mean of the Negative Binomial distribution.

### 3.4 Ensemble Machine Learning Techniques

**Objective:**  
To improve the predictive accuracy of the model by applying advanced
machine learning techniques, specifically Gradient Boosting Trees and
Random Forests.

**Key Methods:**  

- **Gradient Boosting Trees (GBM):** GBM was an ensemble technique that
  built models sequentially, where each new model corrected errors made
  by the previous ones. The general principle was to minimize the loss
  function (e.g., Mean Squared Error) by iteratively fitting a model to
  the residual errors of previous models:

$$
  F_{m+1}(x) = F_m(x) + h_m(x)
  $$

where $h_m(x)$ was the new model fitted to the residuals of the previous
model $F_m(x)$.

- **Random Forest:** Random Forest was another ensemble method that
  aggregates the predictions of multiple decision trees, each built on a
  bootstrap sample of the data. It helped reduce overfitting by
  averaging multiple deep trees, each trained on different parts of the
  data:

  $$
    \hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
    $$

  where $T_b(x)$ was the prediction of the $b$-th tree, and $B$ was the
  number of trees.

### 3.5 Model Evaluation and Improvement

**Objective:**  
To evaluate the performance of the developed models in predicting the
2016 Olympic medal counts and suggest improvements for future
predictions.

**Key Methods:**

- **Evaluation Metrics:**  
  The models were evaluated using Root Mean Square Error (RMSE) and Mean
  Absolute Error (MAE), which measured the differences between the
  predicted and actual medal counts:

$$
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  $$

$$
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  $$

- **Comparison of Models:**  
  The performance of Poisson, Negative Binomial, Zero-Inflated, Gradient
  Boosting Trees, and Random Forest models were compared using the above
  metrics. The model with the lowest RMSE and MAE was considered the
  most accurate.

- **Model Improvement:** Based on the evaluation, improvements may
  include further tuning of hyperparameters, incorporating additional
  variables, or applying more sophisticated ensemble methods to enhance
  predictive performance.

## 4. Exploratory Data Analysis

### 4.1 Data Preparation

The data preparation process was a crucial step in ensuring the
integrity and reliability of the subsequent analysis. The following key
tasks were undertaken:

#### 4.1.1 Loading and Selecting Relevant Variables:

We began by loading the dataset and focusing on the total number of
medals as the dependent variable. We carefully selected relevant
predictor variables, eliminating those that were not necessary for our
analysis.

#### 4.1.2 Handling Missing Values:

To ensure the completeness and accuracy of the dataset, we addressed
missing values by implementing specific imputation strategies. For
countries with missing GDP data, we manually imputed these values based
on reliable external sources. For instance, Afghanistan’s GDP in 2000
was set to 3532 million USD, Cuba’s GDP in 2016 was set to 91370 million
USD, and the Syrian Arab Republic’s GDP in 2016 was set to 12377 million
USD. These imputations were critical to maintaining the consistency and
integrity of the dataset.

Additionally, for missing BMI values, we calculated the mean BMI across
the entire dataset and used this average to fill in the gaps. This
approach ensured that the dataset was complete and free from missing
entries, which is essential for accurate and unbiased analysis. By
applying these methods, we prepared the dataset to be fully utilized in
the subsequent analytical processes.

#### 4.1.3 Creating Subsets for Each Olympic Year:

The dataset was then organized into subsets corresponding to each
Olympic year (2000, 2004, 2008, 2012, and 2016). This approach allowed
for a structured comparison across different years and facilitated the
development of models based on historical data.

#### 4.1.4 Training and Testing Data Preparation:

For model evaluation, the data from the 2016 Olympics was set aside as
the test dataset. The data from previous Olympic years (2000, 2004,
2008, and 2012) were combined to create a comprehensive training
dataset. This separation was critical for testing the predictive
accuracy of our models based on historical trends.

#### 4.1.5 Final Data Checks:

A final check was conducted to ensure that no missing values remained in
the training dataset. This step confirmed the readiness of the data for
further analysis, guaranteeing that the models could be developed
without concerns about data quality.

### 4.2 Correlation Analysis

We conducted a correlation analysis to understand the relationships
between the variables.

<img src="report30_files/figure-gfm/unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

The correlation analysis revealed that several factors were closely
related to the total number of medals a country won at the Olympics.
First, there was a very strong positive correlation between GDP and
total medals (correlation coefficient of 0.76), indicating that
economically stronger countries tended to win more medals. The number of
athletes competing showed an even more significant correlation with
total medals (correlation coefficient of 0.89), suggesting that having
more athletes contributed to a higher medal count. Additionally, being
the host country was also strongly correlated with total medals
(correlation coefficient of 0.66), indicating that host nations often
won more medals, possibly due to home advantage and greater
participation. In contrast, the correlation between population and total
medals was moderate (correlation coefficient of 0.42), while other
variables, such as whether a country was a former Soviet state, a
communist state, a Muslim-majority country, a one-party state, or its
average BMI, showed relatively weak correlations with total medals.
These findings suggested that GDP, the number of athletes, and host
status were the most critical factors in predicting a country’s total
medal count.

### 4.3 Visualize categorical variables

Regarding the categorical variables, the plots indicated that host
countries, communist countries, and one-party states tended to win a
higher number of medals. However, it was important to note that there
were only three one-party countries, with China accounting for a
substantial portion of the medals, which significantly skewed the
boxplot.

<img src="report30_files/figure-gfm/unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

### 4.4 Independence Tests for Categorical Variables

We used chi-squared and Fisher’s exact tests to check the independence
of categorical variables, which helped in ensuring that no redundant
variables are included in the model. The table of independence analysis
was shown in the figure below

<div id="ahfmlznpfn" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#ahfmlznpfn table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
&#10;#ahfmlznpfn thead, #ahfmlznpfn tbody, #ahfmlznpfn tfoot, #ahfmlznpfn tr, #ahfmlznpfn td, #ahfmlznpfn th {
  border-style: none;
}
&#10;#ahfmlznpfn p {
  margin: 0;
  padding: 0;
}
&#10;#ahfmlznpfn .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;#ahfmlznpfn .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}
&#10;#ahfmlznpfn .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}
&#10;#ahfmlznpfn .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}
&#10;#ahfmlznpfn .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}
&#10;#ahfmlznpfn .gt_column_spanner_outer:first-child {
  padding-left: 0;
}
&#10;#ahfmlznpfn .gt_column_spanner_outer:last-child {
  padding-right: 0;
}
&#10;#ahfmlznpfn .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}
&#10;#ahfmlznpfn .gt_spanner_row {
  border-bottom-style: hidden;
}
&#10;#ahfmlznpfn .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}
&#10;#ahfmlznpfn .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}
&#10;#ahfmlznpfn .gt_from_md > :first-child {
  margin-top: 0;
}
&#10;#ahfmlznpfn .gt_from_md > :last-child {
  margin-bottom: 0;
}
&#10;#ahfmlznpfn .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}
&#10;#ahfmlznpfn .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#ahfmlznpfn .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}
&#10;#ahfmlznpfn .gt_row_group_first td {
  border-top-width: 2px;
}
&#10;#ahfmlznpfn .gt_row_group_first th {
  border-top-width: 2px;
}
&#10;#ahfmlznpfn .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#ahfmlznpfn .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_first_summary_row.thick {
  border-top-width: 2px;
}
&#10;#ahfmlznpfn .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#ahfmlznpfn .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}
&#10;#ahfmlznpfn .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#ahfmlznpfn .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}
&#10;#ahfmlznpfn .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#ahfmlznpfn .gt_left {
  text-align: left;
}
&#10;#ahfmlznpfn .gt_center {
  text-align: center;
}
&#10;#ahfmlznpfn .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
&#10;#ahfmlznpfn .gt_font_normal {
  font-weight: normal;
}
&#10;#ahfmlznpfn .gt_font_bold {
  font-weight: bold;
}
&#10;#ahfmlznpfn .gt_font_italic {
  font-style: italic;
}
&#10;#ahfmlznpfn .gt_super {
  font-size: 65%;
}
&#10;#ahfmlznpfn .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}
&#10;#ahfmlznpfn .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}
&#10;#ahfmlznpfn .gt_indent_1 {
  text-indent: 5px;
}
&#10;#ahfmlznpfn .gt_indent_2 {
  text-indent: 10px;
}
&#10;#ahfmlznpfn .gt_indent_3 {
  text-indent: 15px;
}
&#10;#ahfmlznpfn .gt_indent_4 {
  text-indent: 20px;
}
&#10;#ahfmlznpfn .gt_indent_5 {
  text-indent: 25px;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <thead>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="Chi.squared.test">Chi.squared.test</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="Fisher.test">Fisher.test</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="Independence">Independence</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.00</td>
<td headers="Fisher.test" class="gt_row gt_right">0.00</td>
<td headers="Independence" class="gt_row gt_left">independent</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.36</td>
<td headers="Fisher.test" class="gt_row gt_right">0.38</td>
<td headers="Independence" class="gt_row gt_left">independent</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.22</td>
<td headers="Fisher.test" class="gt_row gt_right">0.14</td>
<td headers="Independence" class="gt_row gt_left">independent</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.29</td>
<td headers="Fisher.test" class="gt_row gt_right">0.24</td>
<td headers="Independence" class="gt_row gt_left">independent</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.54</td>
<td headers="Fisher.test" class="gt_row gt_right">0.45</td>
<td headers="Independence" class="gt_row gt_left">independent</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.00</td>
<td headers="Fisher.test" class="gt_row gt_right">0.00</td>
<td headers="Independence" class="gt_row gt_left">associated</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.04</td>
<td headers="Fisher.test" class="gt_row gt_right">0.02</td>
<td headers="Independence" class="gt_row gt_left">independent</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.06</td>
<td headers="Fisher.test" class="gt_row gt_right">0.04</td>
<td headers="Independence" class="gt_row gt_left">independent</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.00</td>
<td headers="Fisher.test" class="gt_row gt_right">0.00</td>
<td headers="Independence" class="gt_row gt_left">independent</td></tr>
    <tr><td headers="Chi.squared.test" class="gt_row gt_right">0.00</td>
<td headers="Fisher.test" class="gt_row gt_right">0.00</td>
<td headers="Independence" class="gt_row gt_left">on the boundary</td></tr>
  </tbody>
  &#10;  
</table>
</div>

The independence analysis between various categorical variables revealed
that most pairs of variables were independent, as indicated by both the
Chi-squared test and Fisher’s exact test results. Specifically,
variables such as “one-party vs. communist,” “one-party vs. Soviet,”
“one-party vs. Muslim,” and “one-party vs. host” all demonstrated
independence. Additionally, the analysis showed that “Soviet
vs. Muslim,” “Soviet vs. host,” “communist vs. Muslim,” and “communist
vs. host” were also independent. However, a significant association was
found between “Soviet vs. communist” countries, suggesting a potential
overlap or relationship between these two categories. The relationship
between “Muslim vs. host” countries appeared to be on the boundary of
significance, indicating a potential but not definitive connection.
Overall, the results highlighted that while most variables were
independent, specific associations, particularly between Soviet and
communist countries, warranted further investigation.

### 4.4 Distribution of Total Medals (2000-2012)

We explored the distribution of the total number of medals won by
countries from 2000 to 2012, as well as specifically in 2012.

<img src="report30_files/figure-gfm/unnamed-chunk-6-1.png" style="display: block; margin: auto auto auto 0;" />

The distribution of total medals won by countries in the Olympic Games
from 2000 to 2012 was highly right-skewed, indicating a significant
disparity in medal counts across nations. Most countries won fewer
medals in these years, and only a few countries won a large number of
medals. This skewed distribution (positive bias) is well suited for
analysis with Poisson regression or negative binomial regression models.
In addition, we can also use the zero expansion model for analysis

## 5. Modeling

We employed various models, including Poisson Regression, Negative
Binomial Regression, and Zero-Inflated models, to predict the total
number of medals won by each country in the 2016 Rio Olympics. Poisson
Regression Poisson regression was used to model count data and was
suitable for predicting the total number of medals.

### 5.1 Poisson Model

During the Poisson regression model fitting process, an initial model
with multiple variables was constructed, followed by model optimization
using stepwise selection. To address the issue of non-independence
between the `muslim` and `host` variables, these variables were removed
one at a time. By comparing the AIC values of the two models after
removing `muslim` and `host`, the model with the `muslim` variable
removed was selected as the final Poisson regression model.

    ## 
    ## Call:
    ## glm(formula = tot ~ gdp + pop + comm + oneparty + altitude + 
    ##     bmi + athletes + host, family = poisson, data = train.data)
    ## 
    ## Coefficients:
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  3.147e+00  3.468e-01   9.074  < 2e-16 ***
    ## gdp          6.165e-08  6.014e-09  10.252  < 2e-16 ***
    ## pop         -9.749e-07  1.022e-07  -9.539  < 2e-16 ***
    ## comm         7.699e-01  4.561e-02  16.879  < 2e-16 ***
    ## oneparty     6.040e-01  1.045e-01   5.783 7.35e-09 ***
    ## altitude    -8.867e-05  4.154e-05  -2.134   0.0328 *  
    ## bmi         -9.412e-02  1.337e-02  -7.039 1.94e-12 ***
    ## athletes     4.786e-03  1.766e-04  27.103  < 2e-16 ***
    ## host         9.833e-01  6.561e-02  14.987  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 8053.7  on 431  degrees of freedom
    ## Residual deviance: 1454.6  on 423  degrees of freedom
    ## AIC: 2594.6
    ## 
    ## Number of Fisher Scoring iterations: 5

The Poisson regression model used in this analysis can be expressed
mathematically as follows:

$$
\text{log}(\mu_i) = \beta_0 + \beta_1 \cdot \text{gdp}_i + \beta_2 \cdot \text{pop}_i + \beta_3 \cdot \text{comm}_i + \beta_4 \cdot \text{oneparty}_i + \beta_5 \cdot \text{altitude}_i + \beta_6 \cdot \text{bmi}_i + \beta_7 \cdot \text{athletes}_i + \beta_8 \cdot \text{host}_i
$$

where: $\mu_i$ was the expected number of medals ($tot_i$) for country
$i$, $\beta_0$ was the intercept, $\beta_1$ was the coefficient for
Gross Domestic Product ($\text{gdp}_i$), $\beta_2$ is the coefficient
for population ($\text{pop}_i$), $\beta_3$ was the coefficient for
whether the country was a communist state ($\text{comm}_i$), $\beta_4$
was the coefficient for whether the country was a one-party state
($\text{oneparty}_i$), $\beta_5$ was the coefficient for the altitude of
the country’s capital city ($\text{altitude}_i$), $\beta_6$ was the
coefficient for the average BMI of the country ($\text{bmi}_i$),
$\beta_7$ was the coefficient for the number of athletes
($\text{athletes}_i$), $\beta_8$ was the coefficient for whether the
country was the host of the Olympics ($\text{host}_i$).

The variance in the Poisson model was equal to the mean:

$$
\text{Var}(Y_i) = \mu_i
$$

### 5.2 Negative Binomial Regression

Negative Binomial regression was applied when the data exhibit
over-dispersion, which was common in count data. The discrete parameter
calculated based on the previous Poisson model is α=3.43, it was obvious
that the negative binomial regression model was more suitable than the
Poisson regression model. This showed that the negative binomial
regression model can not only better capture the variance in the data,
but also improved the accuracy of the prediction when analyzing such
over-discrete counting data. During the Poisson regression model fitting
process, an initial model with multiple variables was constructed,then
all the non-significant variables were removed and the final negative
binomial regression model was constructed with the remaining variables.

    ## 
    ## Call:
    ## glm.nb(formula = tot ~ comm + bmi + athletes + host, data = train.data, 
    ##     init.theta = 1.965347002, link = log)
    ## 
    ## Coefficients:
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  3.7086586  0.7109766   5.216 1.83e-07 ***
    ## comm         0.6923442  0.1034736   6.691 2.22e-11 ***
    ## bmi         -0.1297672  0.0275347  -4.713 2.44e-06 ***
    ## athletes     0.0083818  0.0005313  15.777  < 2e-16 ***
    ## host         0.4474269  0.1693866   2.641  0.00826 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for Negative Binomial(1.9653) family taken to be 1)
    ## 
    ##     Null deviance: 1613.82  on 431  degrees of freedom
    ## Residual deviance:  517.47  on 427  degrees of freedom
    ## AIC: 2125.1
    ## 
    ## Number of Fisher Scoring iterations: 1
    ## 
    ## 
    ##               Theta:  1.965 
    ##           Std. Err.:  0.243 
    ## 
    ##  2 x log-likelihood:  -2113.126

The negative binomial regression model used can be expressed
mathematically as follows:

$$
\text{log}(\mu_i) = \beta_0 + \beta_1 \cdot \text{comm}_i + \beta_2 \cdot \text{bmi}_i + \beta_3 \cdot \text{athletes}_i + \beta_4 \cdot \text{host}_i
$$

where: $\mu_i$ was the expected number of medals ($tot_i$) for country
$i$, $\beta_0$ was the intercept, $\beta_1$ was the coefficient for
whether the country was a communist state ($\text{comm}_i$), $\beta_2$
was the coefficient for the average BMI of the country ($\text{bmi}_i$),
$\beta_3$ was the coefficient for the number of athletes
($\text{athletes}_i$), $\beta_4$ was the coefficient for whether the
country was the host of the Olympics ($\text{host}_i$).

The variance of the model was given by:

$$
\text{Var}(Y_i) = \mu_i + \frac{\mu_i^2}{\theta}
$$

where $\theta$ was the dispersion parameter, estimated in this case to
be approximately 1.965.

This model was used to account for overdispersion in the count data,
where the variance exceeded the mean. By using the log link function,
the model estimated the logarithm of the expected count (total medals)
as a linear combination of the predictor variables.

### 5.3 Zero-Inflated Models

Zero-Inflated models were used when there were an excess number of zeros
in the count data, which can lead to bias in standard Poisson or
Negative Binomial models. Zero-Inflated Poisson Model

#### 5.3.1 Zero-Inflated Poisson Model

#### 5.3.2 Zero-Inflated Negative Binomial Model

We predicted the number of medals for 2016 using each model and evaluate
the performance based on RMSE and MAE.

The analysis involved fitting two zero-inflated models: a Zero-Inflated
Poisson (ZIP) model and a Zero-Inflated Negative Binomial (ZINB) model,
both of which predict the total number of medals. The ZIP model included
`comm`, `host`, and `bmi` as predictors in the count model and
`athletes` as a predictor in the zero-inflation model. Similarly, the
ZINB model used `comm` and `host` as predictors in the count model, with
`athletes` in the zero-inflation part.

After fitting both models, we compared their Akaike Information
Criterion (AIC) values to assess model performance. The ZINB model had a
smaller AIC compared to the ZIP model, indicating that the ZINB model
provides a better fit for the data by accounting for overdispersion and
excess zeros more effectively.

The Zero-Inflated Negative Binomial (ZINB) model used in this analysis
can be expressed mathematically as follows:

##### Count Model (Negative Binomial Component):

$$
\text{log}(\mu_i) = \beta_0 + \beta_1 \cdot \text{comm}_i + \beta_2 \cdot \text{host}_i
$$

where: $\mu_i$ was the expected number of medals ($tot_i$) for country
$i$, $\beta_0$ was the intercept, - $\beta_1$ was the coefficient for
whether the country was a communist state ($\text{comm}_i$), $\beta_2$
was the coefficient for whether the country is the host of the Olympics
($\text{host}_i$).

##### Zero-Inflation Model (Logistic Regression Component):

The probability of excess zeros was modeled using a logistic regression:

$$
\text{logit}(p_i) = \gamma_0 + \gamma_1 \cdot \text{athletes}_i
$$

where: $p_i$ was the probability that country $i$ had an excess zero
(i.e., no medals), $\gamma_0$ was the intercept for the zero-inflation
model, $\gamma_1$ was the coefficient for the number of athletes
($\text{athletes}_i$).

##### Combined ZINB Model:

The combined Zero-Inflated Negative Binomial model was:

$$
P(Y_i = 0) = p_i + (1 - p_i) \cdot \left(\frac{1}{1 + \alpha \mu_i}\right)^{1/\alpha}
$$

$$
P(Y_i = y) = (1 - p_i) \cdot \frac{\Gamma(y + 1/\alpha)}{\Gamma(1/\alpha) \cdot y!} \cdot \left(\frac{1}{1 + \alpha \mu_i}\right)^{1/\alpha} \cdot \left(\frac{\alpha \mu_i}{1 + \alpha \mu_i}\right)^y, \quad y > 0
$$

where $\alpha$ was the dispersion parameter of the Negative Binomial
distribution, and $\Gamma(\cdot)$ represents the Gamma function.

This model accounted for overdispersion in the count data and the
presence of excess zeros by combining a negative binomial count model
with a logistic regression model for zero inflation.

## 6. Prediction and Evaluation

We used test.data to make predictions for the three models respectively,
and calculated their RMSE and MAE for comparison.

As shown in the figure, the previous analysis of the model performance
comparison table highlighted that Poisson Regression, despite having the
lowest RMSE (6.15) and MAE (3.58), might not have been the most
appropriate model due to the presence of overdispersion in the data.
Overdispersion occurred when the variance in the data exceeded the mean,
which could have led to inefficiencies and biases in the Poisson model.
This was evident from the relatively high RMSE and MAE values observed
in the Negative Binomial Regression (RMSE: 13.52, MAE: 5.60) and Zero
Inflated Regression (RMSE: 11.89, MAE: 5.42) models, which were
typically designed to handle such issues but still did not outperform
the Poisson model.

## 6. Improvements

In the previous analysis, we utilized Poisson regression, Negative
Binomial regression, and Zero-Inflated models to predict the total
number of medals a country might win at the Olympics. While these models
performed well in handling count data, particularly in addressing
overdispersion and a high number of zero counts—where the Negative
Binomial and Zero-Inflated models provided better fits—they have
limitations in capturing complex nonlinear relationships and
higher-order interactions between variables. To further enhance
predictive accuracy, we introduced advanced machine learning techniques,
namely Gradient Boosting Trees and Random Forests. These methods
automatically capture nonlinear relationships between variables, are
well-suited for handling high-dimensional data, and effectively reduce
the risk of overfitting by averaging the results of multiple decision
trees or through iterative optimization and weighted adjustments.
Additionally, they exhibit strong robustness when dealing with noisy
data or outliers. By employing these techniques, we aim to further
refine our predictions of Olympic medal counts, thereby improving the
accuracy and reliability of our models

### 6.1 Improvement with Gradient Boosting Trees

To improve the predictive performance of the model, we implemented a
Gradient Boosting Machine (GBM). Gradient Boosting Trees are an ensemble
learning technique that builds models sequentially, each new model
attempting to correct the errors of the previous ones. This method is
particularly effective for handling non-linear relationships and
interactions between variables.

1.  **Model Training**:
    - We trained a GBM model using the training dataset, including
      variables such as `gdp`, `pop`, `comm`, `oneparty`, `altitude`,
      `muslim`, `bmi`, `athletes`, and `host`.
    - The model was configured with 500 trees (`n.trees = 500`), a
      shrinkage rate of 0.05, and an interaction depth of 6.
2.  **Predictions**:
    - We used the trained GBM model to predict the target variable `tot`
      on the test dataset.
3.  **Model Evaluation**:
    - We calculated the Root Mean Square Error (RMSE) and Mean Absolute
      Error (MAE) to assess the model’s performance on the test data.

### 6.2 Random Forest

Random Forests are another ensemble learning method that aggregates the
predictions of multiple decision trees to improve accuracy and control
overfitting.

1.  **Model Training**:
    - We built a Random Forest model using the same predictor variables
      as the GBM model.
    - The model was configured with 500 trees (`ntree = 500`).
2.  **Predictions**:
    - The trained Random Forest model was used to predict the target
      variable `tot` on the test dataset.
3.  **Model Evaluation**:
    - Similar to the GBM model, we calculated the RMSE and MAE to
      evaluate the performance of the Random Forest model on the test
      data.

## 7. Results Comparison

Finally, we compared the performance of all models using RMSE and MAE to
identify the most accurate model for predicting the total number of
medals.

<div id="dwpzqonjel" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#dwpzqonjel table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
&#10;#dwpzqonjel thead, #dwpzqonjel tbody, #dwpzqonjel tfoot, #dwpzqonjel tr, #dwpzqonjel td, #dwpzqonjel th {
  border-style: none;
}
&#10;#dwpzqonjel p {
  margin: 0;
  padding: 0;
}
&#10;#dwpzqonjel .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}
&#10;#dwpzqonjel .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}
&#10;#dwpzqonjel .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}
&#10;#dwpzqonjel .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}
&#10;#dwpzqonjel .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}
&#10;#dwpzqonjel .gt_column_spanner_outer:first-child {
  padding-left: 0;
}
&#10;#dwpzqonjel .gt_column_spanner_outer:last-child {
  padding-right: 0;
}
&#10;#dwpzqonjel .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}
&#10;#dwpzqonjel .gt_spanner_row {
  border-bottom-style: hidden;
}
&#10;#dwpzqonjel .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}
&#10;#dwpzqonjel .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}
&#10;#dwpzqonjel .gt_from_md > :first-child {
  margin-top: 0;
}
&#10;#dwpzqonjel .gt_from_md > :last-child {
  margin-bottom: 0;
}
&#10;#dwpzqonjel .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}
&#10;#dwpzqonjel .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#dwpzqonjel .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}
&#10;#dwpzqonjel .gt_row_group_first td {
  border-top-width: 2px;
}
&#10;#dwpzqonjel .gt_row_group_first th {
  border-top-width: 2px;
}
&#10;#dwpzqonjel .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#dwpzqonjel .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_first_summary_row.thick {
  border-top-width: 2px;
}
&#10;#dwpzqonjel .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#dwpzqonjel .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}
&#10;#dwpzqonjel .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#dwpzqonjel .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}
&#10;#dwpzqonjel .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}
&#10;#dwpzqonjel .gt_left {
  text-align: left;
}
&#10;#dwpzqonjel .gt_center {
  text-align: center;
}
&#10;#dwpzqonjel .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
&#10;#dwpzqonjel .gt_font_normal {
  font-weight: normal;
}
&#10;#dwpzqonjel .gt_font_bold {
  font-weight: bold;
}
&#10;#dwpzqonjel .gt_font_italic {
  font-style: italic;
}
&#10;#dwpzqonjel .gt_super {
  font-size: 65%;
}
&#10;#dwpzqonjel .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}
&#10;#dwpzqonjel .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}
&#10;#dwpzqonjel .gt_indent_1 {
  text-indent: 5px;
}
&#10;#dwpzqonjel .gt_indent_2 {
  text-indent: 10px;
}
&#10;#dwpzqonjel .gt_indent_3 {
  text-indent: 15px;
}
&#10;#dwpzqonjel .gt_indent_4 {
  text-indent: 20px;
}
&#10;#dwpzqonjel .gt_indent_5 {
  text-indent: 25px;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <thead>
    <tr class="gt_heading">
      <td colspan="3" class="gt_heading gt_title gt_font_normal" style>Model Performance Comparison</td>
    </tr>
    <tr class="gt_heading">
      <td colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style>Comparison of RMSE and MAE across different models</td>
    </tr>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="Model">Model</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="Root Mean Square Error (RMSE)">Root Mean Square Error (RMSE)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" scope="col" id="Mean Absolute Error (MAE)">Mean Absolute Error (MAE)</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers="Model" class="gt_row gt_left">Poisson Regression</td>
<td headers="RMSE" class="gt_row gt_right">6.15</td>
<td headers="MAE" class="gt_row gt_right">3.58</td></tr>
    <tr><td headers="Model" class="gt_row gt_left">Negative Binomial Regression</td>
<td headers="RMSE" class="gt_row gt_right">13.52</td>
<td headers="MAE" class="gt_row gt_right">5.60</td></tr>
    <tr><td headers="Model" class="gt_row gt_left">Zero Inflated Regression</td>
<td headers="RMSE" class="gt_row gt_right">11.89</td>
<td headers="MAE" class="gt_row gt_right">5.42</td></tr>
    <tr><td headers="Model" class="gt_row gt_left">Gradient Boosting Machine</td>
<td headers="RMSE" class="gt_row gt_right">7.63</td>
<td headers="MAE" class="gt_row gt_right">3.60</td></tr>
    <tr><td headers="Model" class="gt_row gt_left">Random Forest</td>
<td headers="RMSE" class="gt_row gt_right">6.12</td>
<td headers="MAE" class="gt_row gt_right">3.20</td></tr>
  </tbody>
  &#10;  
</table>
</div>

As shown in the figure, the RMSE and MAE predicted by random forest
model are the smallest, so random forest is the model with the best
prediction effect.

## 8. Conclusions

1.Based on the correlation analysis and the summary results from various
models, we can conclude that GDP, the number of athletes, hosting the
event, and being a communist country are factors associated with the
total number of medals won by each country. Given that the dataset
includes only three socialist countries, with China being the only one
to win a significant number of medals, Therefore, comm cannot be a
factor in determining the total number of medals.

2.We used Poisson regression, Negative Binomial regression, and
Zero-Inflated models to predict each country’s Olympic performance in
2016, taking into account the data characteristics and distribution.
Despite the presence of overdispersion in the data, the Poisson
regression model provided better predictive accuracy compared to the
other models.

3.To improve predicting performance, we addressed the limitations of the
initial models by using Gradient Boosting Trees and Random Forest
models. Among these, the Random Forest model provided the best
predictive accuracy, leading to a significant enhancement in prediction
performance.

## 9. Possible improvements

- Applying a log transformation of gdp, pop, altitude and bmi.
- Using a mixed-effects model allows to incorporate both fixed and
  random effect
