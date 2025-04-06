###############################################################
##### How to Audit Your Gender Pay Gap:
##### An Employers Guide Using Python
#####
##### by Andrew Chamberlain, Ph.D.
#####
##### Original: April 2025
#####
##### Contact:
#####   Web: www.andrewchamberlain.com
#####   Email: andrew.chamberlain@gmail.com
###############################################################

# Load Python libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# Turn off scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)

# Load data
data = pd.read_csv("https://glassdoor.box.com/shared/static/beukjzgrsu35fqe59f7502hruribd5tt.csv")


#############################
# Data Cleaning and Prep.
#############################

# Create five employee age bins
data['age_bin'] = 0
data.loc[data['age'] < 25, 'age_bin'] = 1  # Below age 25
data.loc[(data['age'] >= 25) & (data['age'] < 35), 'age_bin'] = 2  # Age 25-34
data.loc[(data['age'] >= 35) & (data['age'] < 45), 'age_bin'] = 3  # Age 35-44
data.loc[(data['age'] >= 45) & (data['age'] < 55), 'age_bin'] = 4  # Age 45-54
data.loc[data['age'] >= 55, 'age_bin'] = 5  # Age 55+

# Create total compensation variable (base pay + bonus)
data['totalPay'] = data['basePay'] + data['bonus']

# Take natural logarithm of compensation variables
data['log_base'] = np.log(data['basePay'])  # Base pay
data['log_total'] = np.log(data['totalPay'])  # Total comp
data['log_bonus'] = np.log(data['bonus'] + 1)  # Incentive pay. Add 1 to allow for log of 0 bonus values

# Create gender dummies (male = 1, female = 0)
data['male'] = (data['gender'] == 'Male').astype(int)  # Male = 1, Female = 0

# Check the structure of the imported data
print(data.info())

# Cast all categorical variables as categories for the regression analysis
data['jobTitle'] = data['jobTitle'].astype('category')
data['gender'] = data['gender'].astype('category')
data['edu'] = data['edu'].astype('category')
data['dept'] = data['dept'].astype('category')


#############################
# Summary Statistics. 
#############################

# Create an overall table of summary statistics for the data
summary_stats = data.describe()
summary_stats.to_html('summary.html')

# Base pay summary stats
summary_base = data.groupby('gender').agg(
    meanBasePay=('basePay', 'mean'),
    medBasePay=('basePay', 'median'),
    cnt=('basePay', 'count')
)
print(summary_base)

# Total pay summary stats
summary_total = data.groupby('gender').agg(
    meanTotalPay=('totalPay', 'mean'),
    medTotalPay=('totalPay', 'median'),
    cnt=('totalPay', 'count')
)
print(summary_total)

# Bonus summary stats
summary_bonus = data.groupby('gender').agg(
    meanBonus=('bonus', 'mean'),
    medBonus=('bonus', 'median'),
    cnt=('bonus', 'count')
)
print(summary_bonus)

# Performance evaluations summary stats
summary_perf = data.groupby('gender').agg(
    meanPerf=('perfEval', 'mean'),
    cnt=('perfEval', 'count')
)
print(summary_perf)

# Departmental distribution of employees
summary_dept = data.groupby(['dept', 'gender']).agg(
    meanTotalPay=('totalPay', 'mean'),
    cnt=('dept', 'count')
).reset_index().sort_values(by=['dept', 'gender'], ascending=[False, True])
print(summary_dept)

# Job title distribution of employees
summary_job = data.groupby(['jobTitle', 'gender']).agg(
    meanTotalPay=('totalPay', 'mean'),
    cnt=('jobTitle', 'count')
).reset_index().sort_values(by=['jobTitle', 'gender'], ascending=[False, True])
print(summary_job)


###########################################################################################
# Model Estimation: OLS with controls. 
# Coefficient on "male" has the interpretation of approximate male pay advantage ("gender pay gap").
###########################################################################################

#############################
# Logarithm of Base Pay
#############################

# No controls. ("unadjusted" pay gap.)
model1 = smf.ols('log_base ~ male', data=data).fit()
print(model1.summary())

# Adding "human capital" controls (performance evals, age and education)
model2 = smf.ols('log_base ~ male + perfEval + C(age_bin) + C(edu)', data=data).fit()
print(model2.summary())

# Adding all controls. ("adjusted" pay gap.)
model3 = smf.ols('log_base ~ male + perfEval + C(age_bin) + C(edu) + C(dept) + seniority + C(jobTitle)', data=data).fit()
print(model3.summary())

# Print log base pay "adjusted" gender pay gap and p-value
logbase_pay_gap = model3.params['male']  # male coefficient for adjusted pay gap
logbase_pay_pvalue = model3.pvalues['male']  # associated p value
print(f"Adjusted gender pay gap (log): {logbase_pay_gap}")
print(f"P-value: {logbase_pay_pvalue}")

# Create a summary table of regression results
models = [model1, model2, model3]
model_names = ['Model 1', 'Model 2', 'Model 3']

# Create basic summary table without the problematic info_dict parameter
results_table = summary_col(models, 
                          float_format='%0.4f',
                          stars=True, 
                          model_names=model_names,
                          regressor_order=['Intercept', 'male', 'perfEval', 'seniority'])

# Add the additional information manually as HTML
control_info = """
<tr><td>Controls:</td><td></td><td></td><td></td></tr>
<tr><td>Education</td><td>No</td><td>Yes</td><td>Yes</td></tr>
<tr><td>Department</td><td>No</td><td>No</td><td>Yes</td></tr>
<tr><td>Job Title</td><td>No</td><td>No</td><td>Yes</td></tr>
"""

html_content = results_table.as_html()
html_content = html_content.replace("</table>", control_info + "</table>")

# Save the regression table to HTML
with open('results.html', 'w') as f:
    f.write(html_content)


#############################
# Results by Department
# (Interaction of male x dept)
# To test for differences by department, examine significance of each "male x dept" coefficient.
# For the gender pay gap by department, add the "male" + "male x dept" coefficients from this model. 
#############################

# All controls with department interaction terms
dept_formula = 'log_base ~ male*C(dept) + perfEval + C(age_bin) + C(edu) + seniority + C(jobTitle)'
dept_results = smf.ols(dept_formula, data=data).fit()
print(dept_results.summary())

# Save results to CSV (similar to tidy() in R)
dept_results_df = pd.DataFrame({
    'term': dept_results.params.index,
    'estimate': dept_results.params.values,
    'std.error': dept_results.bse.values,
    'statistic': dept_results.tvalues.values,
    'p.value': dept_results.pvalues.values
})
dept_results_df.to_csv('dept_clean.csv', index=False)

# Save the results to HTML (equivalent to stargazer)
with open('dept.html', 'w') as f:
    f.write(dept_results.summary().as_html())


#############################
# Results by Job Title 
# (Interaction of male x job title) 
# To test for differences by job title, examine significance of each "male x job title" coefficient.
# For the gender pay gap by job title, add the "male" + "male x job title" coefficients from this model. 
#############################

# All controls with job title interaction terms
job_formula = 'log_base ~ male*C(jobTitle) + perfEval + C(age_bin) + C(edu) + seniority + C(dept)'
job_results = smf.ols(job_formula, data=data).fit()
print(job_results.summary())

# Save results to CSV (similar to tidy() in R)
job_results_df = pd.DataFrame({
    'term': job_results.params.index,
    'estimate': job_results.params.values,
    'std.error': job_results.bse.values,
    'statistic': job_results.tvalues.values,
    'p.value': job_results.pvalues.values
})
job_results_df.to_csv('job_clean.csv', index=False)

# Save the results to HTML (equivalent to stargazer)
with open('job.html', 'w') as f:
    f.write(job_results.summary().as_html())

# For additional analysis via Oaxaca-Blinder decomposition, the Python package 'oaxaca' can be used
# It can be installed via: pip install git+https://github.com/iangow/oaxaca.git

##### END ######
