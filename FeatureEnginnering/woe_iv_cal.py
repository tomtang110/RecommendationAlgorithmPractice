import pandas as pd
import numpy as np
def iv_woe(data, var,target, show_woe=False):
    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    # Extract Column Names
    cols = data.columns

    # Run WOE and IV on all the independent variables


        # Calculate the number of events in each group (bin)
    d = data.groupby(var, as_index=False).agg({target: ["count", "sum"]})
    d.columns = ['Cutoff', 'N', 'Events']

     # Calculate % of events in each group.
    d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()

    # Calculate the non events in each group.
    d['Non-Events'] = d['N'] - d['Events']
    # Calculate % of non events in each group.
    d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()

    # Calculate WOE by taking natural log of division of % of non-events and % of events
    d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
    d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
    d.insert(loc=0, column='Variable', value=var)
    print("Information value of " + var + " is " + str(round(d['IV'].sum(), 6)))
    temp = pd.DataFrame({"Variable": [var], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
    newDF = pd.concat([newDF, temp], axis=0)
    woeDF = pd.concat([woeDF, d], axis=0)

    # Show WOE Table
    if show_woe == True:
        print(d)
    return newDF, woeDF