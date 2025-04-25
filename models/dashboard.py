# models/dashboard.py   (v4-patch)  🔧
# ───────────────────────────────────────────────────────────────
from pathlib import Path
import re, numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objs as go

BASE = Path(__file__).resolve().parent        # .../models

# ─── Theme colours ─────────────────────────────────────────────
THEME = {
    "Overview"     : ( 70, 70, 70),
    "Goals"        : ( 28, 55,107),   # navy
    "Assists"      : (178, 34, 34),   # crimson
    "Clean Sheets" : (255,215,  0),   # gold
    "Cards"        : ( 34,139, 34),   # forest-green
    "Total Points" : (106, 76,147),   # purple
    "Optimiser"    : ( 30,144,255)    # dodger-blue
}
rgba = lambda rgb,a: f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"

def fade(sty,rgb,n=5):
    fades=[1,.8,.6,.4,.2]
    def _row(r):
        idx=r.name
        color=rgba(rgb,fades[idx-1]) if idx<=n else None
        return [f'background-color:{color};color:white' if color else '' for _ in r]
    return sty.apply(_row,axis=1)

# ─── file helper ───────────────────────────────────────────────
def newest(pattern):
    for folder in (BASE, BASE.parent):
        files=list(folder.glob(pattern))
        if files:
            files.sort(key=lambda p:int(re.search(r"GW(\d+)",str(p)).group(1)),
                       reverse=True)
            return files[0]
    return None

PAT = {
    "Goals"        :"GW*_Predicted_goals_with_fixtures.csv",
    "Assists"      :"GW*_Predicted_assists.csv",
    "Clean Sheets" :"GW*_Predicted_[cC]lean_[sS]heets.csv",
    "Cards"        :"GW*_Predicted_cards.csv"
}

# ─── Streamlit set-up ──────────────────────────────────────────
st.set_page_config("FPL Dashboard", layout="wide")
header=Path(__file__).with_name("static")/"robs.png"
if header.exists(): st.image(str(header))
st.title("🧮  FPL Prediction Dashboard")

TAB = st.sidebar.radio("Choose view",
        ["Overview","Goals","Assists","Clean Sheets","Cards",
         "Total Points","Optimiser"])
rgb=THEME[TAB]

# ─── reusable render fns ───────────────────────────────────────
def render_table(df,title,fmt=None):
    df=df.head(25).reset_index(drop=True); df.index+=1
    sty=df.style.format(fmt or {}).pipe(fade,rgb)
    st.subheader(title)
    st.dataframe(sty,height=min(30+len(df)*28,640),use_container_width=True)

def scatter(df,title):
    fig=px.scatter(df,x='FixDiff',y='Prob',
                   hover_data=['Player','Team','Opp(H/A)'],
                   color_discrete_sequence=[rgba(rgb,.8)])
    fig.update_layout(height=460,title=title,
                      xaxis_title='Fixture difficulty',
                      yaxis_title='Probability')
    st.plotly_chart(fig,use_container_width=True)

goal_pts=lambda p:5 if p=='MID' else 4
cs_pts  =lambda p:4 if p in('GK','DEF') else 0
ASSIST_PTS,CARD_PTS=3,-1

# ───────────────────── Overview ────────────────────────────────
if TAB=="Overview":
    st.header("Welcome – what do these numbers mean?")
    st.markdown("""
**Probabilities** are produced by four Random-Forest models (Goals,
Assists, Clean-sheets, Cards).  
**Total Points** converts them to expected FPL points.  
**Optimiser** picks the best £100 m squad (≤3 per club) via linear
programming (PuLP).

| Column | Meaning | Points if it happens |
|--------|---------|----------------------|
| GoalsProb | Chance the player scores | 4 (FWD/DEF) or 5 (MID) |
| AstProb   | Chance of an assist | 3 pts |
| CSProb    | Chance of a clean-sheet | 4 pts (GK/DEF) |
| CardProb  | Chance of a yellow/red | −1 pt |
| Minutes   | Avg. minutes last 3 GWs | ≥60 ⇒ +2 pts |
""")

# ───────────────────── Metric tabs ─────────────────────────────
def metric_tab(pattern,col,title):
    path=newest(pattern)
    if path is None:
        st.warning(f"No {title} file found."); return
    full=pd.read_csv(path); gw=re.search(r"GW(\d+)",str(path)).group(1)
    teams=sorted(full.team.unique())
    full=full[ full.team.isin(st.sidebar.multiselect("Filter team(s)",teams,teams)) ]

    df=full[['name','team','opponent_name','was_home',col,'fixture_difficulty']]\
        .rename(columns={'name':'Player','team':'Team','opponent_name':'Opponent',
                         col:'Prob','fixture_difficulty':'FixDiff'})
    df['Opp(H/A)']=df.apply(lambda r:f"{r.Opponent}({'H' if r.was_home else 'A'})",axis=1)
    df.drop(columns=['Opponent','was_home'],inplace=True)
    df.sort_values('Prob',ascending=False,inplace=True)

    render_table(df,f"{title} – GW{gw}",{'Prob':'{:.2f}'})
    scatter(df,f'Fixture difficulty vs {title.lower()} probability')

if TAB=="Goals":
    metric_tab(PAT["Goals"],"predicted_goals","Goals")
elif TAB=="Assists":
    metric_tab(PAT["Assists"],"predicted_assists","Assists")
elif TAB=="Cards":
    metric_tab(PAT["Cards"],"predicted_card_prob","Cards")
elif TAB=="Clean Sheets":
    path=newest(PAT["Clean Sheets"])
    if path is None: st.warning("No Clean-sheet file."); st.stop()
    raw=pd.read_csv(path); gw=re.search(r"GW(\d+)",str(path)).group(1)
    if 'clean_sheet_probability' in raw.columns:
        raw=raw.rename(columns={'clean_sheet_probability':'predicted_cs_prob'})
    teams=sorted(raw.team.unique())
    raw=raw[ raw.team.isin(st.sidebar.multiselect("Filter team(s)",teams,teams)) ]
    df=raw[['name','team','opponent_name','was_home','predicted_cs_prob','fixture_difficulty']]\
        .rename(columns={'name':'Player','team':'Team','opponent_name':'Opponent',
                         'predicted_cs_prob':'Prob','fixture_difficulty':'FixDiff'})
    df['Opp(H/A)']=df.apply(lambda r:f"{r.Opponent}({'H' if r.was_home else 'A'})",axis=1)
    df.drop(columns=['Opponent','was_home'],inplace=True)
    df.sort_values('Prob',ascending=False,inplace=True)

    render_table(df,f"Clean-sheets – GW{gw}",{'Prob':'{:.2f}'})
    scatter(df,'Fixture difficulty vs Clean-sheet probability')

    team_bar=df.groupby('Team')['Prob'].max().sort_values(ascending=False).head(12)
    fig=go.Figure(go.Bar(y=team_bar.index,x=team_bar.values,orientation='h',
                         marker_color=[rgba(rgb,.8)]*len(team_bar)))
    fig.update_layout(height=460,title="Highest CS chance per team",
                      yaxis={'autorange':'reversed'})
    st.plotly_chart(fig,use_container_width=True)

# ───────────────── Total Points ────────────────────────────────
elif TAB=="Total Points":
    paths=[newest(PAT[p]) for p in ("Goals","Assists","Clean Sheets","Cards")]
    if None in paths: st.warning("Missing prediction files."); st.stop()
    g,a,c,k=[pd.read_csv(p) for p in paths]
    gw=re.search(r"GW(\d+)",str(paths[0])).group(1)
    g=g.rename(columns={'name':'Player','team':'Team','predicted_goals':'GoalsProb',
                        'position':'Position','roll3_minutes':'Minutes','minutes':'Minutes'})
    for col,d in {'GoalsProb':0,'Minutes':60,'Position':'MID'}.items(): g[col]=g.get(col,d)
    a=a.rename(columns={'name':'Player','team':'Team','predicted_assists':'AstProb'})
    c=c.rename(columns={'player_name':'Player','name':'Player','team':'Team',
                        'clean_sheet_probability':'CSprob','predicted_cs_prob':'CSprob'})
    k=k.rename(columns={'name':'Player','team':'Team','predicted_card_prob':'CardProb'})
    merged=(g[['Player','Team','Position','GoalsProb','Minutes']]
            .merge(a[['Player','Team','AstProb']],on=['Player','Team'],how='outer')
            .merge(c[['Player','Team','CSprob']],on=['Player','Team'],how='outer')
            .merge(k[['Player','Team','CardProb']],on=['Player','Team'],how='outer'))
    merged.fillna({'GoalsProb':0,'AstProb':0,'CSprob':0,'CardProb':0,'Minutes':60,'Position':'MID'},inplace=True)
    merged['ExpPts']=(merged.GoalsProb*merged.Position.map(goal_pts)+
                      merged.AstProb*ASSIST_PTS+
                      merged.CSprob*merged.Position.map(cs_pts)+
                      np.where(merged.Minutes>=60,2,0)+merged.CardProb*CARD_PTS)
    teams=sorted(merged.Team.unique())
    merged=merged[ merged.Team.isin(st.sidebar.multiselect("Filter team(s)",teams,teams)) ]
    merged.sort_values('ExpPts',ascending=False,inplace=True)
    render_table(merged[['Player','Team','Position','ExpPts','GoalsProb','AstProb','CSprob','CardProb']],
                 f"Expected points – GW{gw}",{'ExpPts':'{:.2f}','GoalsProb':'{:.2f}',
                                              'AstProb':'{:.2f}','CSprob':'{:.2f}','CardProb':'{:.2f}'})
    fig=px.scatter(merged.head(100),x='GoalsProb',y='AstProb',size='ExpPts',color='ExpPts',
                   color_continuous_scale='Viridis',hover_data=['Player','Team'],
                   title='Goals vs Assists (bubble = ExpPts)')
    fig.update_layout(height=500); st.plotly_chart(fig,use_container_width=True)

# ───────────────── Optimiser ───────────────────────────────────
elif TAB == "Optimiser":
    squad_path = newest("GW*_OptimalSquad.csv")
    if squad_path is None:
        st.warning("Run optimise_team.py first."); st.stop()

    xi_path = squad_path.with_name(squad_path.name.replace("Squad", "XI"))
    squad   = pd.read_csv(squad_path)
    xi      = (pd.read_csv(xi_path)
           .rename(columns={'name':'Player',    # capital P for display
                            'Position':'position'}))  # normalise case
    gw      = re.search(r"GW(\d+)", str(squad_path)).group(1)

    st.subheader(f"Optimal squad – GW{gw}")
    st.dataframe(
        squad.style.format({'price': '{:.1f}', 'ExpPts': '{:.2f}'}),
        height=min(30 + len(squad) * 28, 640),
        use_container_width=True
    )

    # ── Build formation layout ─────────────────────────────────
    lines = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}   # row order
    pitch_y = {row: 100 - 20 * row for row in lines.values()}  # y-coords
# x-coordinates for the centre of each slot
    pitch_x = {1: 50, 50: 50}    # include explicit key 50 for GK lookup

    # spread defenders / mids / fwds evenly across pitch width 10–90
    for pos, slots in [('DEF', 5), ('MID', 5), ('FWD', 3)]:
        n = len(xi[xi.position == pos])
        pitch_x.update({f"{pos}{i}": np.linspace(10, 90, n)[i]
                        for i in range(n)})
    # assign coordinates
    coords = []
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        players = xi[xi.position == pos].reset_index(drop=True)
        for i, r in players.iterrows():
            x_key = 50 if pos == 'GK' else f"{pos}{i}"
            coords.append({
                'Player' : r.Player,
                'team'   : r.team,
                'pos'    : pos,
                'ExpPts' : r.ExpPts,
                'x'      : pitch_x[x_key],
                'y'      : pitch_y[lines[pos]]
            })
    form_df = pd.DataFrame(coords)

    # detect numerical formation for title
    form = "-".join(str(len(xi[xi.position == p])) for p in ['DEF', 'MID', 'FWD'])

    pos_colour = {'GK': '#1f77b4', 'DEF': '#2ca02c',
                  'MID': '#ff7f0e', 'FWD': '#d62728'}

    # ── Pitch figure ───────────────────────────────────────────
    fig = go.Figure()
    # full-pitch rectangle  (put it *below* points)
    fig.add_shape(
    type="rect",
    x0=0, y0=0, x1=100, y1=100,
    fillcolor="#3b9957", line=dict(width=0),
    layer="below"              
    )

    # centre line
    fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50,
                  line=dict(color="white", width=2))

    # players
    for _, r in form_df.iterrows():
        fig.add_trace(go.Scatter(
        x=[r.x], y=[r.y],
        mode='markers+text',
        marker=dict(
            symbol='square',
            size=54,
            color='white',                       # jersey fill
            line=dict(color=pos_colour[r.pos], width=6)  # outline by role
        ),
        text=r.Player.split()[-1],               # surname
        textfont=dict(color='black', size=11),
        textposition='middle center',
        hovertemplate=f"<b>{r.Player}</b><br>{r.team}<br>"
                      f"{r.pos} – {r.ExpPts:.2f} pts<extra></extra>"
    ))


    fig.update_layout(
        height=460, width=700,
        title=f"Optimal XI – {form} formation",
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False, range=[0, 100]),
        plot_bgcolor="#3b9957",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI bars (unchanged) ───────────────────────────────────
    tot_price = xi.price.sum()
    tot_pts   = xi.ExpPts.sum()
    vfm       = tot_pts / tot_price

    kpi = pd.DataFrame({
        'Metric': ['Total Price (£m)', 'Expected Points', 'Pts per £m'],
        'Value' : [tot_price,          tot_pts,           vfm]
    })
    fig2 = px.bar(
        kpi, x='Value', y='Metric', orientation='h', text='Value',
        color_discrete_sequence=[rgba(rgb, .8)]
    )
    fig2.update_traces(texttemplate='%{x:.2f}', textposition='inside')
    fig2.update_layout(
        height=260, margin=dict(l=40, r=20, t=30, b=25),
        xaxis_title='', yaxis_title='',
        yaxis=dict(categoryorder='total ascending')
    )
    st.plotly_chart(fig2, use_container_width=True)
