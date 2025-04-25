# models/dashboard.py
# ───────────────────────────────────────────────────────────────
# Streamlit dashboard for FPL predictions
#
#  Tabs
#   • Goals
#   • Assists
#   • Clean Sheets
#   • Total Points (Goals + Assists + CS + minutes)
#
# Author: Dylan Chai
# ───────────────────────────────────────────────────────────────

from pathlib import Path
import re, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[0]

# ── helper ------------------------------------------------------
def newest_file(pattern: str, base: Path) -> Path | None:
    """Return newest GW file that matches pattern (or None)."""
    files = list(base.glob(pattern))
    if not files:
        return None
    files.sort(
        key=lambda p: int(re.search(r"GW(\d+)", p.name).group(1)),
        reverse=True
    )
    return files[0]

def load_latest(pattern: str):
    p = newest_file(pattern, BASE_DIR)
    return (None, None) if p is None else (pd.read_csv(p), p)

def goal_pts(pos): return 5 if pos == 'MID' else 4
def cs_pts(pos):   return 4 if pos in ('GK', 'DEF') else 0
ASSIST_PTS = 3   # 3 pts per assist

# patterns
PAT = {
    "Goals"        : "GW*_Predicted_goals_with_fixtures.csv",
    "Assists"      : "GW*_Predicted_assists.csv",
    "Clean Sheets" : "GW*_Predicted_[cC]lean_[sS]heets.csv",
    "Cards" : "GW*_Predicted_cards.csv",

    "Total Points" : None               # composite
}

# ── page setup --------------------------------------------------
st.set_page_config(page_title="FPL Prediction Dashboard", layout="wide")
st.title("🧮  FPL Prediction Dashboard")
choice = st.sidebar.radio("Select model", list(PAT.keys()))

# ───────────────────────── GOALS TAB ───────────────────────────
if choice == "Goals":
    path = newest_file(PAT["Goals"], BASE_DIR)
    if path is None:
        st.warning("No Goals file found.")
        st.stop()

    df  = pd.read_csv(path)
    gw  = re.search(r"GW(\d+)", path.name).group(1)

    teams = sorted(df.team.unique())
    df = df[df.team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))]

    df = df[['name','team','opponent_name','was_home','predicted_goals']]\
           .rename(columns={'name':'Player','team':'Team',
                            'opponent_name':'Opponent','predicted_goals':'Prob'})
    df['Opp(H/A)'] = df.apply(lambda r:f"{r.Opponent} ({'H' if r.was_home else 'A'})",axis=1)
    df.drop(columns=['Opponent','was_home'], inplace=True)
    df = df.sort_values('Prob', ascending=False).reset_index(drop=True)
    df.index += 1

    st.subheader(f"Goals – GW{gw}")
    st.dataframe(df.style.format({'Prob':'{:.2f}'}), use_container_width=True)
    st.plotly_chart(
        px.bar(df.head(12), x='Prob', y=df.head(12).index,
               orientation='h', labels={'index':'Rank'},
               title="Top 12 goal probabilities").update_yaxes(autorange='reversed'),
        use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       f"Goals_GW{gw}.csv", "text/csv")
    st.stop()

# ──────────────────────── ASSISTS TAB ──────────────────────────
elif choice == "Assists":
    path = newest_file(PAT["Assists"], BASE_DIR)
    if path is None:
        st.warning("No Assists file found.")
        st.stop()

    df  = pd.read_csv(path)
    gw  = re.search(r"GW(\d+)", path.name).group(1)

    teams = sorted(df.team.unique())
    df = df[df.team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))]

    df = df[['name','team','opponent_name','was_home','predicted_assists']]\
           .rename(columns={'name':'Player','team':'Team',
                            'opponent_name':'Opponent','predicted_assists':'Prob'})
    df['Opp(H/A)'] = df.apply(lambda r:f"{r.Opponent} ({'H' if r.was_home else 'A'})",axis=1)
    df.drop(columns=['Opponent','was_home'], inplace=True)
    df = df.sort_values('Prob', ascending=False).reset_index(drop=True)
    df.index += 1

    st.subheader(f"Assists – GW{gw}")
    st.dataframe(df.style.format({'Prob':'{:.2f}'}), use_container_width=True)
    st.plotly_chart(
        px.bar(df.head(12), x='Prob', y=df.head(12).index,
               orientation='h', labels={'index':'Rank'},
               title="Top 12 assist probabilities").update_yaxes(autorange='reversed'),
        use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       f"Assists_GW{gw}.csv", "text/csv")
    st.stop()

# ─────────────────── CLEAN-SHEET TAB ───────────────────────────
elif choice == "Clean Sheets":
    path = newest_file(PAT["Clean Sheets"], BASE_DIR)
    if path is None:
        st.warning("No Clean-sheet file found.")
        st.stop()

    raw = pd.read_csv(path)
    gw  = re.search(r"GW(\d+)", path.name).group(1)

    # legacy vs new schema
    if 'clean_sheet_probability' in raw.columns:
        df = raw[['player_name','team','opponent','clean_sheet_probability']]\
              .rename(columns={'player_name':'Player','team':'Team',
                               'opponent':'Opp(H/A)','clean_sheet_probability':'Prob'})
    else:
        df = raw[['name','team','opponent_name','predicted_cs_prob']]\
              .rename(columns={'name':'Player','team':'Team',
                               'opponent_name':'Opp(H/A)','predicted_cs_prob':'Prob'})

    teams = sorted(df.Team.unique())
    df = df[df.Team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))]
    df = df.sort_values('Prob', ascending=False).reset_index(drop=True)
    df.index += 1

    st.subheader(f"Clean-sheets – GW{gw}")
    st.dataframe(df.style.format({'Prob':'{:.2f}'}), use_container_width=True)
    st.plotly_chart(
        px.bar(df.head(12), x='Prob', y=df.head(12).index,
               orientation='h', labels={'index':'Rank'},
               title="Top 12 CS probabilities").update_yaxes(autorange='reversed'),
        use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       f"CS_GW{gw}.csv", "text/csv")
    st.stop()

# ─────────────────── TOTAL POINTS TAB ──────────────────────────
else:   # Total Points
    g_df, g_path = load_latest(PAT["Goals"])
    a_df, a_path = load_latest(PAT["Assists"])
    c_df, c_path = load_latest(PAT["Clean Sheets"])
    if None in (g_path, a_path, c_path):
        st.warning("Need Goals, Assists **and** Clean-sheet files.")
        st.stop()

    gw = re.search(r"GW(\d+)", g_path.name).group(1)

    # harmonise columns
    g_df = g_df.rename(columns={
    'name': 'Player', 'team': 'Team',
    'predicted_goals': 'GoalsProb',
    'position': 'Position',
    'roll3_minutes': 'Minutes',   # new-style
    'minutes': 'Minutes'          # old-style
})
    if 'Minutes' not in g_df.columns:
        g_df['Minutes'] = 60

    a_df = a_df.rename(columns={'name':'Player','team':'Team',
                                'predicted_assists':'AstProb'})
    c_df = c_df.rename(columns={'player_name':'Player','name':'Player',
                                'team':'Team',
                                'clean_sheet_probability':'CSprob',
                                'predicted_cs_prob':'CSprob'})

    merged = (g_df[['Player','Team','Position','GoalsProb','Minutes']]
              .merge(a_df[['Player','Team','AstProb']], on=['Player','Team'], how='outer')
              .merge(c_df[['Player','Team','CSprob']], on=['Player','Team'], how='outer'))

    merged['Minutes']   = merged['Minutes'].fillna(60)
    merged['GoalsProb'] = merged['GoalsProb'].fillna(0)
    merged['AstProb']   = merged['AstProb'].fillna(0)
    merged['CSprob']    = merged['CSprob'].fillna(0)
    merged['Position']  = merged['Position'].fillna('MID')

    merged['PtsGoals'] = merged.apply(lambda r: r.GoalsProb*goal_pts(r.Position), axis=1)
    merged['PtsAst']   = merged['AstProb']  * ASSIST_PTS
    merged['PtsCS']    = merged.apply(lambda r: r.CSprob  * cs_pts(r.Position), axis=1)
    merged['PtsMin']   = np.where(merged['Minutes'] >= 60, 2, 0)
    merged['ExpPts']   = merged[['PtsGoals','PtsAst','PtsCS','PtsMin']].sum(axis=1)

    teams = sorted(merged.Team.unique())
    merged = merged[
        merged.Team.isin(st.sidebar.multiselect("Filter team(s)", teams, teams))
    ]
    merged = merged.sort_values('ExpPts', ascending=False).reset_index(drop=True)
    merged.index += 1

    disp = merged[['Player','Team','Position','ExpPts',
                   'GoalsProb','AstProb','CSprob','PtsMin']]
    st.subheader(f"Total expected FPL points – GW{gw}")
    st.dataframe(disp.style.format({'ExpPts':'{:.2f}',
                                    'GoalsProb':'{:.2f}','AstProb':'{:.2f}','CSprob':'{:.2f}'}),
                 use_container_width=True)

    st.plotly_chart(
        px.bar(disp.head(12), x='ExpPts', y=disp.head(12).index,
               orientation='h', labels={'index':'Rank'},
               title="Top 12 total-point projections").update_yaxes(autorange='reversed'),
        use_container_width=True)
    st.download_button("Download CSV", disp.to_csv(index=False).encode(),
                       f"TotalPoints_GW{gw}.csv", "text/csv")
