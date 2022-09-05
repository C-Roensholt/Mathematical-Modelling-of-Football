import numpy as np

# progressive passes (25% closer to goal)
def calculate_progressive_passes(df):
    df['beginning'] = np.sqrt(np.square(120 - df['x']) + np.square(40 - df['y']))
    df['end'] = np.sqrt(np.square(120 - df['end_x']) + np.square(40 - df['end_y']))
    df.reset_index(inplace=True, drop=True)
    # Get progressive passes
    df['progressive'] = [(df['end'][x]) / (df['beginning'][x]) < .75 for x in range(len(df['beginning']))]
    # Filter for progressive passes
    df = df.loc[df['progressive'] == True].reset_index(drop=True)
    return df

# deep completions
def calculate_deep_completions(df):
    df['initialDistancefromgoal'] = np.sqrt(((120 - df['x'])**2) + ((40 - df['y'])**2))
    df['finalDistancefromgoal'] = np.sqrt(((120 - df['end_x'])**2) + ((40 - df['end_y'])**2))

    df['deepCompletion'] = (np.where(((df['finalDistancefromgoal'] <= (21.87)) &
                                      (df['initialDistancefromgoal'] >= (21.87))), True, False))
    df_deep_completion = df[df["deepCompletion"]==True]
    return df_deep_completion