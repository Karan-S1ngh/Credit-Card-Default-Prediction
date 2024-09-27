# Predicting whether the customer will default on his/her credit card


### Problem Description
This project is aimed at predicting the case of customers default payments.

### Data Description
##### Attribute Information
Data has one target variable and 23 variables as explanatory variables: 
default payment (Yes = 1, No = 0) - Target Variable

Education : 1 = Graduate School, 2 = university, 3 = high school, 4 = others
Marriage status : 1 = married, 2 = single, 3 = other

Pay1 - Pay6 : History of past payment (Pay0 - repayment status in Sept, 2005: Pay1 - repayment status in Aug, 2005; .... Pay6 - repayment status in April, 2005)

Measurement : 0 = payed on time, -1 = payed duly, 1 = one month delay, 2 = 2 month delay ...

BILL_AMT1 - BILL_AMT6 : Amount of Bill statement. (BILL_AMT1 - Bill statement in Sept, 2005; ... BILL_AMT6 - Bill statement in April, 2005)

PAY_AMT1 - PAY_AMT6 : Amount of previous payment (PAY_AMT1 - amount payed in Sept, 2005; ... PAY_AMT6 - amount payed in April, 2005)


##### Detailed Information
PAY_1: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months,8=payment delay for eight - months, 9=payment delay for nine months and above)
PAY_2: Repayment status in August, 2005 (scale same as above)
PAY_3: Repayment status in July, 2005 (scale same as above)
PAY_4: Repayment status in June, 2005 (scale same as above)
PAY_5: Repayment status in May, 2005 (scale same as above)
PAY_6: Repayment status in April, 2005 (scale same as above)
BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
