import math
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

def draw_arrow_with_shrink(ax, x, y, end_x, end_y, lw, line_color, alpha, dist_delta=2.5):
    dist = math.hypot(end_x - x, end_y - y)
    angle = math.atan2(end_y - y, end_x - x)
    upd_end_x = x + (dist - dist_delta) * math.cos(angle)
    upd_end_y = y + (dist - dist_delta) * math.sin(angle)
    upd_x = end_x - (dist - dist_delta * 1.2) * math.cos(angle)
    upd_y = end_y - (dist - dist_delta * 1.2) * math.sin(angle)
    ax.annotate('', xy=(upd_end_x, upd_end_y), xytext=(upd_x, upd_y), zorder=1,
                arrowprops=dict(linewidth=lw, color=line_color, alpha=alpha,
                                headwidth=15, headlength=15))