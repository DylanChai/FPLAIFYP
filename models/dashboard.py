# models/dashboard.py   (v3 – per-tab colours & CS team chart)
# ───────────────────────────────────────────────────────────────
from pathlib import Path
import re, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go

BASE_DIR = Path(__file__).resolve().parents[0]

# ---------- colour definitions ---------------------------------
THEME = {
    "Goals"        : ( 28, 55,107),   # navy
    "Clean Sheets"      : (178, 34, 34),   # crimson
    "Cards" : (255,215,  0),   # gold
    "Assists"        : ( 34,139, 34),   # forest-green
    "Total Points" : (106, 76,147)    # purple (for bubble chart)
}

def faded_rgba(rgb, alpha):   # helper
    r,g,b = rgb
    return f"rgba({r},{g},{b},{alpha})"

def fade_table(styler, rgb, n=5):
    fades = [1,.8,.6,.4,.2]
    def row_colour(row):
        idx = row.name
        if idx < n:
            return [f'background-color:{faded_rgba(rgb, fades[idx])};color:white']*len(row)
        return ['']*len(row)
    return styler.apply(row_colour, axis=1)

# ---------- helpers --------------------------------------------
def newest_file(pattern: str):
    files = list(BASE_DIR.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: int(re.search(r"GW(\d+)", p.name).group(1)),
               reverse=True)
    return files[0]

def load_latest(pattern):
    p = newest_file(pattern)
    return (None, None) if p is None else (pd.read_csv(p), p)

def goal_pts(pos): return 5 if pos == 'MID' else 4
def cs_pts(pos):   return 4 if pos in ('GK','DEF') else 0
ASSIST_PTS, CARD_PTS = 3, -1

PAT = {
    "Goals"        : "GW*_Predicted_goals_with_fixtures.csv",
    "Assists"      : "GW*_Predicted_assists.csv",
    "Clean Sheets" : "GW*_Predicted_[cC]lean_[sS]heets.csv",  # case-flex
    "Cards"        : "GW*_Predicted_cards.csv",
    "Total Points" : None
}

# ---------- Streamlit layout -----------------------------------
st.set_page_config(page_title="FPL Prediction Dashboard", layout="wide")
header = Path(__file__).with_name("static") / "robs.png"
if header.exists(): st.image(str(header))
st.title("🧮  FPL Prediction Dashboard")

choice = st.sidebar.radio("Select model", list(PAT.keys()))
rgb    = THEME[choice]

def render_table(df, title, fmt=None):
    df = df.reset_index(drop=True).head(25)
    df.index += 1
    sty = df.style
    if fmt: sty = sty.format(fmt)
    sty = fade_table(sty, rgb)
    st.subheader(title)
    st.dataframe(sty, height=min(30+df.shape[0]*28, 640),
                 use_container_width=True)

# ---------- common scatter chart -------------------------------
def scatter_fixture(df, x='FixDiff', y='Prob', title=''):
    fig = px.scatter(df, x=x, y=y,
                     hover_data=['Player','Team','Opp(H/A)'],
                     color_discrete_sequence=[faded_rgba(rgb,0.8)])
    fig.update_layout(title=title, height=460,
                      xaxis_title='Fixture difficulty',
                      yaxis_title='Probability')
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────── Goals tab ───────────────────────────
if choice == "Goals":
    path = newest_file(PAT["Goals"])
    if path is None:
        st.warning("No Goals file found."); st.stop()
    full = pd.read_csv(path); gw = re.search(r"GW(\d+)", path.name).group(1)

    teams = sorted(full.team.unique())
    full  = full[full.team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))]

    df = full[['name','team','opponent_name','was_home',
               'predicted_goals','fixture_difficulty']]\
         .rename(columns={'name':'Player','team':'Team','opponent_name':'Opponent',
                          'predicted_goals':'Prob','fixture_difficulty':'FixDiff'})
    df['Opp(H/A)'] = df.apply(lambda r:f"{r.Opponent}({'H' if r.was_home else 'A'})", axis=1)
    df.drop(columns=['Opponent','was_home'], inplace=True)
    df.sort_values('Prob', ascending=False, inplace=True)

    render_table(df, f"Goals – GW{gw}", {'Prob':'{:.2f}'})
    scatter_fixture(df, title='Fixture difficulty vs Goal probability')

# ───────────────────────── Assists tab ─────────────────────────
elif choice == "Assists":
    path = newest_file(PAT["Assists"])
    if path is None: st.warning("No Assists file found."); st.stop()
    full = pd.read_csv(path); gw = re.search(r"GW(\d+)", path.name).group(1)

    teams = sorted(full.team.unique())
    full  = full[full.team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))]

    df = full[['name','team','opponent_name','was_home',
               'predicted_assists','fixture_difficulty']]\
         .rename(columns={'name':'Player','team':'Team','opponent_name':'Opponent',
                          'predicted_assists':'Prob','fixture_difficulty':'FixDiff'})
    df['Opp(H/A)'] = df.apply(lambda r:f"{r.Opponent}({'H' if r.was_home else 'A'})", axis=1)
    df.drop(columns=['Opponent','was_home'], inplace=True)
    df.sort_values('Prob', ascending=False, inplace=True)

    render_table(df, f"Assists – GW{gw}", {'Prob':'{:.2f}'})
    scatter_fixture(df, title='Fixture difficulty vs Assist probability')

# ─────────────────────── Clean-sheet tab ───────────────────────
elif choice == "Clean Sheets":
    path = newest_file(PAT["Clean Sheets"])
    if path is None: st.warning("No Clean-sheet file found."); st.stop()
    raw = pd.read_csv(path); gw = re.search(r"GW(\d+)", path.name).group(1)

    if 'clean_sheet_probability' in raw.columns:  # harmonise
        raw.rename(columns={'player_name':'name','opponent':'opponent_name',
                            'clean_sheet_probability':'predicted_cs_prob'}, inplace=True)

    teams = sorted(raw.team.unique())
    raw   = raw[raw.team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))]

    df = raw[['name','team','opponent_name','was_home',
              'predicted_cs_prob','fixture_difficulty']]\
         .rename(columns={'name':'Player','team':'Team','opponent_name':'Opponent',
                          'predicted_cs_prob':'Prob','fixture_difficulty':'FixDiff'})
    df['Opp(H/A)'] = df.apply(lambda r:f"{r.Opponent}({'H' if r.was_home else 'A'})", axis=1)
    df.drop(columns=['Opponent','was_home'], inplace=True)
    df.sort_values('Prob', ascending=False, inplace=True)

    render_table(df, f"Clean-sheets – GW{gw}", {'Prob':'{:.2f}'})
    scatter_fixture(df, title='Fixture difficulty vs Clean-sheet probability')

    # team-level bar chart (highest player prob per team)
    team_bar = df.groupby('Team')['Prob'].max().sort_values(ascending=False).head(12)
    fig = go.Figure(go.Bar(
        x=team_bar.values, y=team_bar.index, orientation='h',
        marker_color=[faded_rgba(rgb, .8)]*len(team_bar)
    ))
    fig.update_layout(title="Best CS chance by team", height=460,
                      yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────── Cards tab ───────────────────────────
elif choice == "Cards":
    path = newest_file(PAT["Cards"])
    if path is None: st.warning("No Cards file found."); st.stop()
    full = pd.read_csv(path); gw = re.search(r"GW(\d+)", path.name).group(1)

    teams = sorted(full.team.unique())
    full  = full[full.team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))]

    df = full[['name','team','opponent_name','was_home',
               'predicted_card_prob','fixture_difficulty']]\
         .rename(columns={'name':'Player','team':'Team','opponent_name':'Opponent',
                          'predicted_card_prob':'Prob','fixture_difficulty':'FixDiff'})
    df['Opp(H/A)'] = df.apply(lambda r:f"{r.Opponent}({'H' if r.was_home else 'A'})", axis=1)
    df.drop(columns=['Opponent','was_home'], inplace=True)
    df.sort_values('Prob', ascending=False, inplace=True)

    render_table(df, f"Cards – GW{gw}", {'Prob':'{:.2f}'})
    scatter_fixture(df, title='Fixture difficulty vs Card probability')

# ───────────────────── Total-Points tab ────────────────────────
else:
    g_df, g_p = load_latest(PAT["Goals"])
    a_df, a_p = load_latest(PAT["Assists"])
    c_df, c_p = load_latest(PAT["Clean Sheets"])
    k_df, k_p = load_latest(PAT["Cards"])
    if None in (g_p, a_p, c_p, k_p): st.warning("Missing files."); st.stop()
    gw = re.search(r"GW(\d+)", g_p.name).group(1)

    # harmonise & defaults
    g_df = g_df.rename(columns={'name':'Player','team':'Team','predicted_goals':'GoalsProb',
                                'position':'Position','roll3_minutes':'Minutes','minutes':'Minutes'})
    for col, default in {'GoalsProb':0,'Minutes':60,'Position':'MID'}.items():
        if col not in g_df.columns: g_df[col] = default
    a_df = a_df.rename(columns={'name':'Player','team':'Team','predicted_assists':'AstProb'})
    c_df = c_df.rename(columns={'player_name':'Player','name':'Player','team':'Team',
                                'clean_sheet_probability':'CSprob','predicted_cs_prob':'CSprob'})
    k_df = k_df.rename(columns={'name':'Player','team':'Team','predicted_card_prob':'CardProb'})

    merged = (g_df[['Player','Team','Position','GoalsProb','Minutes']]
              .merge(a_df[['Player','Team','AstProb']],  on=['Player','Team'], how='outer')
              .merge(c_df[['Player','Team','CSprob']],  on=['Player','Team'], how='outer')
              .merge(k_df[['Player','Team','CardProb']],on=['Player','Team'], how='outer'))
    merged.fillna({'GoalsProb':0,'AstProb':0,'CSprob':0,'CardProb':0,'Minutes':60,'Position':'MID'}, inplace=True)

    merged['PtsGoals'] = merged.apply(lambda r: r.GoalsProb*goal_pts(r.Position), axis=1)
    merged['PtsAst']   = merged['AstProb']   * ASSIST_PTS
    merged['PtsCS']    = merged.apply(lambda r: r.CSprob   * cs_pts(r.Position), axis=1)
    merged['PtsMin']   = np.where(merged['Minutes'] >= 60, 2, 0)
    merged['PtsCard']  = merged['CardProb']  * CARD_PTS
    merged['ExpPts']   = merged[['PtsGoals','PtsAst','PtsCS','PtsMin','PtsCard']].sum(axis=1)

    teams = sorted(merged.Team.unique())
    merged = merged[merged.Team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))]
    merged.sort_values('ExpPts', ascending=False, inplace=True)

    show = ['Player','Team','Position','ExpPts','GoalsProb','AstProb','CSprob','CardProb','PtsMin']
    render_table(merged[show], f"Total expected FPL points – GW{gw}",
                 {'ExpPts':'{:.2f}','GoalsProb':'{:.2f}','AstProb':'{:.2f}',
                  'CSprob':'{:.2f}','CardProb':'{:.2f}'})

    top = merged.head(100)
    fig = px.scatter(top, x='GoalsProb', y='AstProb', size='ExpPts', color='ExpPts',
                     color_continuous_scale='Viridis', hover_data=['Player','Team'],
                     title='Goals vs Assists vs ExpPts (bubble size)')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
