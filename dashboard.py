import os
import ast
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime

print("🚀 Loading Instagram Dashboard...")

# ====================================
# HELPER FUNCTIONS (from original)
# ====================================

def parse_list_like(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        v = val.strip()
        if v.startswith('[') and v.endswith(']'):
            try:
                out = ast.literal_eval(v)
                return out if isinstance(out, list) else []
            except Exception:
                return []
    return []

def calculate_growth_metrics(df):
    if len(df) < 2 or df['timestamp'].isna().all():
        return {
            'posts_growth': 0,
            'engagement_growth': 0,
            'likes_growth': 0,
            'trend': 'stable'
        }

    df_sorted = df.sort_values('timestamp')
    mid_point = len(df_sorted) // 2
    first_half = df_sorted.iloc[:mid_point]
    second_half = df_sorted.iloc[mid_point:]

    first_avg_eng = first_half['engagement'].mean()
    second_avg_eng = second_half['engagement'].mean()

    eng_growth = ((second_avg_eng - first_avg_eng) / first_avg_eng * 100) if first_avg_eng > 0 else 0

    first_weeks = (first_half['timestamp'].max() - first_half['timestamp'].min()).days / 7
    second_weeks = (second_half['timestamp'].max() - second_half['timestamp'].min()).days / 7

    first_posts_pw = len(first_half) / first_weeks if first_weeks > 0 else 0
    second_posts_pw = len(second_half) / second_weeks if second_weeks > 0 else 0

    posts_growth = ((second_posts_pw - first_posts_pw) / first_posts_pw * 100) if first_posts_pw > 0 else 0

    trend = 'growing' if eng_growth > 5 else ('declining' if eng_growth < -5 else 'stable')

    return {
        'posts_growth': posts_growth,
        'engagement_growth': eng_growth,
        'likes_growth': 0,
        'trend': trend
    }

# ====================================
# LOAD AND PREPARE DATA
# ====================================

df_raw = pd.read_csv('instagram_cleaned.csv', low_memory=False)

# Parse timestamp and convert to PST/PDT
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='coerce', utc=True)
if df_raw['timestamp'].notna().any():
    # Convert UTC to PST/PDT
    df_raw['timestamp'] = df_raw['timestamp'].dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)

# Parse lists
df_raw['hashtags'] = df_raw['hashtags'].apply(parse_list_like)
df_raw['mentions'] = df_raw['mentions'].apply(parse_list_like)

# Time fields
df_raw['weekday'] = df_raw['timestamp'].dt.day_name()
df_raw['hour_utc'] = df_raw['timestamp'].dt.hour
df_raw['hour_12hr'] = df_raw['timestamp'].dt.strftime('%I %p')
df_raw['week'] = df_raw['timestamp'].dt.to_period('W').astype(str)
df_raw['date'] = df_raw['timestamp'].dt.date

# Engagement calculations
df_raw['engagement_weighted'] = df_raw['likes'] + df_raw['comments'] + (df_raw['views'] * 0.1)

if len(df_raw) > 0 and df_raw['engagement'].max() > 0:
    df_raw['engagement_score'] = (df_raw['engagement'] / df_raw['engagement'].max() * 100)
else:
    df_raw['engagement_score'] = 0

# Count hashtags and mentions
df_raw['hashtag_count'] = df_raw['hashtags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df_raw['mention_count'] = df_raw['mentions'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df_raw['caption_length'] = df_raw['caption_text'].str.len()
df_raw['caption_snippet'] = df_raw['caption_text'].str.slice(0, 140)

# ====================================
# FILTER OPTIONS
# ====================================

unique_post_types = sorted(df_raw['post_type'].dropna().unique())
unique_owners = sorted([u for u in df_raw['ownerUsername'].unique() if u and str(u).strip()])

def explode_hashtags(df):
    rows = []
    for _, r in df.iterrows():
        for h in (r['hashtags'] if isinstance(r['hashtags'], list) else []):
            rows.append({
                'hashtag': str(h).lower(),
                'engagement': r['engagement'],
                'post_type': r.get('post_type', '')
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['hashtag','engagement','post_type'])

hashtag_options = sorted(explode_hashtags(df_raw)['hashtag'].value_counts().head(200).index.tolist())
WEEKDAY_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# Date defaults
if df_raw['timestamp'].notna().any():
    _end_default = df_raw['timestamp'].max().date()
    _start_default = (df_raw['timestamp'].max() - pd.Timedelta(days=90)).date()
else:
    _start_default = None
    _end_default = None

# ====================================
# FILTER FUNCTION
# ====================================

def filter_df(df, start_date, end_date, post_types, hashtags, keyword, owner_usernames):
    f = df.copy()
    
    # Filter by account
    if owner_usernames:
        f = f[f['ownerUsername'].isin(owner_usernames)]
    
    # Filter by date
    if start_date is not None:
        f = f[f['timestamp'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        f = f[f['timestamp'] < (pd.to_datetime(end_date) + pd.Timedelta(days=1))]
    
    # Filter by post type
    if post_types:
        f = f[f['post_type'].isin(post_types)]
    
    # Filter by hashtags
    if hashtags:
        hs_lower = [str(hh).lower() for hh in hashtags]
        def _has_any(hs):
            if not isinstance(hs, list):
                return False
            base = [str(x).lower() for x in hs]
            return any(h in base for h in hs_lower)
        f = f[f['hashtags'].apply(_has_any)]
    
    # Filter by keyword
    if keyword and str(keyword).strip():
        kw = str(keyword).strip().lower()
        f = f[f['caption_text'].str.lower().str.contains(kw, na=False)]
    
    return f

# ====================================
# CREATE APP
# ====================================

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = 'Instagram Performance Dashboard'

# ====================================
# LAYOUT COMPONENTS
# ====================================

# Controls with Account Filter
controls = dbc.Card([
    html.Div([
        html.Div([
            html.Label('Filter by Account(s)', style={'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='owner-username-select',
                options=[{'label': f'@{u}', 'value': u} for u in unique_owners],
                multi=True,
                placeholder='Select account(s) - leave empty for all',
                style={'width': '100%'}
            )
        ], className='col-md-12 mb-3'),
    ], className='row', style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '15px'}),
    
    html.Div([
        html.Div([
            html.Label('Date range', style={'fontWeight': 'bold'}),
            dcc.DatePickerRange(
                id='date-range',
                start_date=_start_default,
                end_date=_end_default,
                display_format='YYYY-MM-DD'
            )
        ], className='col-md-3 mb-2'),
        html.Div([
            html.Label('Post types', style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='post-type', options=[{'label': t, 'value': t} for t in unique_post_types], 
                        multi=True, placeholder='All types')
        ], className='col-md-3 mb-2'),
        html.Div([
            html.Label('Hashtags', style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='hashtag-select', options=[{'label': '#'+h, 'value': h} for h in hashtag_options], 
                        multi=True, placeholder='Any hashtag')
        ], className='col-md-4 mb-2'),
        html.Div([
            html.Label('Keyword', style={'fontWeight': 'bold'}),
            dcc.Input(id='keyword-input', type='text', placeholder='Search caption...', className='form-control')
        ], className='col-md-2 mb-2'),
    ], className='row'),
], body=True)

# KPI card helper
def kpi_card(id_, title, subtitle_id=None):
    card_body = [html.H4(id=id_, className='card-title mb-0', style={'color': '#1f77b4'})]
    if subtitle_id:
        card_body.append(html.Small(id=subtitle_id, className='text-muted'))
    return dbc.Card([
        dbc.CardHeader(title, style={'fontSize': '14px', 'fontWeight': 'bold'}),
        dbc.CardBody(card_body)
    ], className='mb-3 shadow-sm')

# KPIs
kpis_row1 = dbc.Row([
    dbc.Col(kpi_card('kpi-posts', 'Total Posts', 'kpi-posts-subtitle'), md=2),
    dbc.Col(kpi_card('kpi-likes', 'Total Likes', 'kpi-likes-subtitle'), md=2),
    dbc.Col(kpi_card('kpi-comments', 'Total Comments', 'kpi-comments-subtitle'), md=2),
    dbc.Col(kpi_card('kpi-views', 'Video Views', 'kpi-views-subtitle'), md=2),
    dbc.Col(kpi_card('kpi-avg-engagement', 'Avg Engagement', 'kpi-avg-engagement-subtitle'), md=2),
    dbc.Col(kpi_card('kpi-engagement-rate', 'Engagement Rate', 'kpi-engagement-rate-subtitle'), md=2),
])

kpis_row2 = dbc.Row([
    dbc.Col(kpi_card('kpi-best-post-type', 'Best Content Type'), md=3),
    dbc.Col(kpi_card('kpi-avg-hashtags', 'Avg Hashtags/Post'), md=3),
    dbc.Col(kpi_card('kpi-posting-frequency', 'Posts per Week'), md=3),
    dbc.Col(kpi_card('kpi-consistency', 'Posting Consistency'), md=3),
])

# All chart sections
charts = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader('Engagement Trend (with Moving Average)', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='ts-engagement-trend', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=6),
    dbc.Col(dbc.Card([
        dbc.CardHeader('Content Type Performance', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='content-type-comparison', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=6),
], className='mb-3')

distribution_charts = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader('Engagement Distribution', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='engagement-distribution', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=6),
    dbc.Col(dbc.Card([
        dbc.CardHeader('Likes vs Comments Correlation', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='likes-comments-correlation', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=6),
], className='mb-3')

heat_and_hash = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader('Best times to post (PST/PDT)', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='heatmap-engagement', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=6),
    dbc.Col(dbc.Card([
        dbc.CardHeader('Top 15 Hashtags by Usage', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='bar-top-hashtags', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=6),
], className='mb-3')

caption_analysis = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader('Caption Length vs Engagement', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='caption-length-analysis', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=6),
    dbc.Col(dbc.Card([
        dbc.CardHeader('Hashtag Count vs Engagement', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='hashtag-count-analysis', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=6),
], className='mb-3')

hash_perf = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader('Top 15 Hashtags by Avg Engagement', style={'fontWeight': 'bold'}),
        dbc.CardBody(dcc.Graph(id='bar-avg-engagement-hashtags', config={'displayModeBar': False}))
    ], className='shadow-sm'), md=12)
], className='mb-3')

performance_tables = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader('🏆 Top 5 Performing Posts', style={'fontWeight': 'bold'}),
        dbc.CardBody(html.Div(id='top-posts-list'))
    ], className='shadow-sm'), md=6),
    dbc.Col(dbc.Card([
        dbc.CardHeader('📉 Bottom 5 Performing Posts', style={'fontWeight': 'bold'}),
        dbc.CardBody(html.Div(id='bottom-posts-list'))
    ], className='shadow-sm'), md=6),
], className='mb-3')

post_table = dbc.Card([
    dbc.CardHeader('Post Browser (click links to open Instagram)', style={'fontWeight': 'bold'}),
    dbc.CardBody([
        dash_table.DataTable(
            id='table-posts',
            columns=[
                {'name': 'Date', 'id': 'date'},
                {'name': 'Account', 'id': 'ownerUsername'},
                {'name': 'Type', 'id': 'post_type'},
                {'name': 'Likes', 'id': 'likes', 'type': 'numeric'},
                {'name': 'Comments', 'id': 'comments', 'type': 'numeric'},
                {'name': 'Views', 'id': 'views', 'type': 'numeric'},
                {'name': 'Engagement', 'id': 'engagement', 'type': 'numeric'},
                {'name': 'Score', 'id': 'engagement_score', 'type': 'numeric'},
                {'name': 'Hashtags', 'id': 'hashtag_count', 'type': 'numeric'},
                {'name': 'Caption', 'id': 'caption_snippet'},
                {'name': 'Link', 'id': 'link', 'presentation': 'markdown'}
            ],
            page_size=15,
            sort_action='native',
            filter_action='native',
            style_cell={'fontSize': '13px', 'textAlign': 'left', 'padding': '8px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
            style_data_conditional=[
                {'if': {'filter_query': '{engagement_score} > 80'}, 'backgroundColor': '#d4edda', 'color': '#155724'},
                {'if': {'filter_query': '{engagement_score} < 20'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'}
            ]
        ),
        html.Button('📥 Download Filtered CSV', id='btn-download', className='btn btn-primary mt-3'),
        dcc.Download(id='download-data')
    ])
], className='shadow-sm')

# ====================================
# APP LAYOUT
# ====================================

app.layout = dbc.Container([
    html.H2('📊 Instagram Performance Dashboard', className='mt-4 mb-2', style={'color': '#1f77b4'}),
    html.P('Interactive analytics for posts, engagement, hashtags, and posting times', className='text-muted mb-4'),
    html.Hr(),
    controls,
    html.Hr(),
    kpis_row1,
    kpis_row2,
    html.Hr(),
    charts,
    html.Hr(),
    distribution_charts,
    html.Hr(),
    heat_and_hash,
    html.Hr(),
    caption_analysis,
    html.Hr(),
    hash_perf,
    html.Hr(),
    performance_tables,
    html.Hr(),
    post_table,
    html.Footer(className='mt-5 mb-3 text-center text-muted', children=[
        html.Small('Built with Plotly Dash • Powered by Sonora AI')
    ])
], fluid=True)

# ====================================
# MAIN CALLBACK - ALL CHARTS AND KPIs
# ====================================

@app.callback(
    [Output('kpi-posts','children'),
     Output('kpi-posts-subtitle','children'),
     Output('kpi-likes','children'),
     Output('kpi-likes-subtitle','children'),
     Output('kpi-comments','children'),
     Output('kpi-comments-subtitle','children'),
     Output('kpi-views','children'),
     Output('kpi-views-subtitle','children'),
     Output('kpi-avg-engagement','children'),
     Output('kpi-avg-engagement-subtitle','children'),
     Output('kpi-engagement-rate','children'),
     Output('kpi-engagement-rate-subtitle','children'),
     Output('kpi-best-post-type','children'),
     Output('kpi-avg-hashtags','children'),
     Output('kpi-posting-frequency','children'),
     Output('kpi-consistency','children'),
     Output('ts-engagement-trend','figure'),
     Output('content-type-comparison','figure'),
     Output('engagement-distribution','figure'),
     Output('likes-comments-correlation','figure'),
     Output('heatmap-engagement','figure'),
     Output('bar-top-hashtags','figure'),
     Output('caption-length-analysis','figure'),
     Output('hashtag-count-analysis','figure'),
     Output('bar-avg-engagement-hashtags','figure'),
     Output('top-posts-list','children'),
     Output('bottom-posts-list','children'),
     Output('table-posts','data')],
    [Input('owner-username-select','value'),
     Input('date-range','start_date'),
     Input('date-range','end_date'),
     Input('post-type','value'),
     Input('hashtag-select','value'),
     Input('keyword-input','value')]
)
def update_dashboard(owner_usernames, start_date, end_date, post_types, hashtag_values, keyword):
    # Filter data
    f = filter_df(df_raw, start_date, end_date, post_types, hashtag_values, keyword, owner_usernames)
    
    # Handle empty data
    if len(f) == 0:
        empty_fig = px.scatter(title='No data available for selected filters')
        return (
            "0", "No data", "0", "No data", "0", "No data", "0", "No data",
            "0", "No data", "0", "No data", "N/A", "0", "0", "0%",
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig,
            [html.P("No data")], [html.P("No data")], []
        )
    
    # ====================================
    # CALCULATE KPIs
    # ====================================
    
    growth = calculate_growth_metrics(f)
    
    total_posts = int(len(f))
    total_likes = int(f['likes'].sum())
    total_comments = int(f['comments'].sum())
    total_views = int(f['views'].sum())
    avg_engagement = float(f['engagement'].mean())
    engagement_rate = (total_likes + total_comments) / total_posts if total_posts > 0 else 0
    
    if len(f) > 0:
        best_type = f.groupby('post_type')['engagement'].mean().idxmax()
        best_type_avg = f.groupby('post_type')['engagement'].mean().max()
    else:
        best_type = "N/A"
        best_type_avg = 0
    
    avg_hashtags = f['hashtag_count'].mean()
    
    if f['timestamp'].notna().any():
        date_range_days = (f['timestamp'].max() - f['timestamp'].min()).days
        posts_per_week = (len(f) / date_range_days * 7) if date_range_days > 0 else 0
    else:
        posts_per_week = 0
    
    # Consistency Score
    if f['timestamp'].notna().any() and len(f) > 1:
        weekly_posts = f.set_index('timestamp').resample('W-MON').size()
        if len(weekly_posts) >= 2:
            active_weeks = (weekly_posts > 0).sum()
            total_weeks = len(weekly_posts)
            coverage_pct = (active_weeks / total_weeks) * 100
            active_posts = weekly_posts[weekly_posts > 0]
            if len(active_posts) > 1 and active_posts.mean() > 0:
                cv = active_posts.std() / active_posts.mean()
                evenness_score = max(0, 100 - (cv * 50))
            else:
                evenness_score = 50
            consistency_score = (coverage_pct * 0.6) + (evenness_score * 0.4)
            consistency_score = round(min(100, max(0, consistency_score)))
        else:
            consistency_score = 0
    else:
        consistency_score = 0
    
    # KPI Subtitles
    posts_subtitle = f"{'↑' if growth['posts_growth'] > 0 else '↓' if growth['posts_growth'] < 0 else '→'} {abs(growth['posts_growth']):.1f}% vs prev period"
    likes_subtitle = f"Avg: {total_likes/total_posts:.0f} per post" if total_posts > 0 else "No data"
    comments_subtitle = f"Avg: {total_comments/total_posts:.1f} per post" if total_posts > 0 else "No data"
    views_subtitle = f"Avg: {total_views/len(f[f['views']>0]):.0f} per video" if len(f[f['views']>0]) > 0 else "No video posts"
    avg_eng_subtitle = f"{'↑' if growth['engagement_growth'] > 0 else '↓' if growth['engagement_growth'] < 0 else '→'} {abs(growth['engagement_growth']):.1f}% trend"
    eng_rate_subtitle = "Per post interaction rate"
    
    # ====================================
    # 1. ENGAGEMENT TREND WITH MOVING AVERAGE
    # ====================================
    
    if f['timestamp'].notna().any() and len(f) > 0:
        ts_df = f.set_index('timestamp').sort_index()
        weekly_eng = ts_df['engagement'].resample('W-MON').sum().reset_index()
        weekly_eng['post_count'] = ts_df['engagement'].resample('W-MON').count().values
        weekly_eng['ma_4w'] = weekly_eng['engagement'].rolling(4, min_periods=1).mean()
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Bar(
            x=weekly_eng['timestamp'],
            y=weekly_eng['engagement'],
            name='Weekly Engagement',
            marker=dict(
                color=weekly_eng['post_count'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Posts", x=1.12)
            ),
            text=weekly_eng['post_count'],
            texttemplate='%{text} posts',
            textposition='outside',
            hovertemplate='<b>Week of %{x|%b %d}</b><br>Engagement: %{y}<br>Posts: %{text}<extra></extra>'
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=weekly_eng['timestamp'],
            y=weekly_eng['ma_4w'],
            mode='lines+markers',
            name='4-Week Trend',
            line=dict(color='#ef4444', width=3, dash='dash'),
            marker=dict(size=6, color='#ef4444', symbol='diamond'),
            hovertemplate='<b>Week of %{x|%b %d}</b><br>4-Week Avg: %{y:.1f}<extra></extra>'
        ))
        
        fig_trend.update_layout(
            margin=dict(l=20, r=80, t=10, b=40),
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', title=None, tickformat='%b %d'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', title='Total Engagement'),
            height=380
        )
    else:
        fig_trend = px.line(title='No timestamp data available')
    
    # ====================================
    # 2. CONTENT TYPE COMPARISON
    # ====================================
    
    if len(f) > 0 and f['post_type'].notna().any():
        content_stats = f.groupby('post_type').agg({
            'engagement': 'mean',
            'post_type': 'count'
        }).rename(columns={'post_type': 'count'}).reset_index()
        
        fig_content = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Avg Engagement', 'Post Count'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        fig_content.add_trace(
            go.Bar(x=content_stats['post_type'], y=content_stats['engagement'], 
                   name='Engagement', marker_color='steelblue'),
            row=1, col=1
        )
        
        fig_content.add_trace(
            go.Pie(labels=content_stats['post_type'], values=content_stats['count'], name='Count'),
            row=1, col=2
        )
        
        fig_content.update_layout(margin=dict(l=10,r=10,t=40,b=10), showlegend=False, height=380)
    else:
        fig_content = px.bar(title='No data')
    
    # ====================================
    # 3. ENGAGEMENT DISTRIBUTION
    # ====================================
    
    if len(f) > 0:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=f['engagement'],
            nbinsx=30,
            name='Engagement',
            marker_color='steelblue'
        ))
        fig_dist.add_vline(
            x=f['engagement'].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {f['engagement'].median():.0f}"
        )
        fig_dist.update_layout(
            margin=dict(l=10,r=10,t=30,b=10),
            xaxis_title='Engagement',
            yaxis_title='Number of Posts',
            height=380
        )
    else:
        fig_dist = px.histogram(title='No data')
    
    # ====================================
    # 4. LIKES VS COMMENTS CORRELATION
    # ====================================
    
    if len(f) > 0:
        f_clean = f[['likes', 'comments', 'post_type', 'engagement', 'caption_snippet']].copy()
        f_clean = f_clean[(f_clean['likes'] >= 0) & (f_clean['comments'] >= 0)]
        
        if len(f_clean) > 1:
            corr = f_clean[['likes', 'comments']].corr().iloc[0, 1]
            f_clean['engagement_size'] = f_clean['engagement'].clip(lower=1)
            
            fig_corr = px.scatter(
                f_clean,
                x='likes',
                y='comments',
                color='post_type',
                size='engagement_size',
                hover_data=['caption_snippet'],
                title=f'Correlation: {corr:.2f}'
            )
            fig_corr.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=380)
        else:
            fig_corr = px.scatter(title='Not enough data for correlation')
    else:
        fig_corr = px.scatter(title='No data')
    
    # ====================================
    # 5. HEATMAP - BEST POSTING TIMES (PST/PDT)
    # ====================================
    
    hm_df = f.copy()
    hm_df['weekday'] = pd.Categorical(hm_df['weekday'], categories=WEEKDAY_ORDER, ordered=True)
    heat = hm_df.pivot_table(index='weekday', columns='hour_utc', values='engagement', aggfunc='mean')
    heat = heat.reindex(WEEKDAY_ORDER)
    
    if heat.shape[1] > 0:
        heat = heat.loc[:, ~heat.columns.isna()]
        if heat.shape[1] > 0:
            cols_sorted = sorted([c for c in heat.columns if pd.notna(c)])
            heat = heat.reindex(cols_sorted, axis=1)
    
    if heat.size > 0 and not heat.isna().all().all():
        # Convert to 12-hour format
        hour_labels = []
        for hour in heat.columns:
            if pd.notna(hour):
                h = int(hour)
                period = 'AM' if h < 12 else 'PM'
                display_hour = h if h <= 12 else h - 12
                display_hour = 12 if display_hour == 0 else display_hour
                hour_labels.append(f"{display_hour} {period}")
            else:
                hour_labels.append('')
        
        fig_heat = px.imshow(
            heat,
            aspect='auto',
            color_continuous_scale='YlOrRd',
            labels=dict(x="Hour (PST/PDT)", y="Day of Week", color="Avg Engagement"),
            x=hour_labels
        )
        fig_heat.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=380)
        fig_heat.update_xaxes(tickangle=-45)
    else:
        fig_heat = px.imshow([[0]], title='No data for heatmap')
    
    # ====================================
    # 6 & 7. HASHTAG CHARTS
    # ====================================
    
    hs_rows = []
    for _, r in f.iterrows():
        hs = r['hashtags'] if isinstance(r['hashtags'], list) else []
        for h in hs:
            hs_rows.append({
                'hashtag': str(h).strip().lower(),
                'engagement': r.get('engagement', 0),
                'post_type': r.get('post_type', '')
            })
    
    if hs_rows:
        hs_df = pd.DataFrame(hs_rows)
        
        # Top hashtags by usage
        top_hs = hs_df['hashtag'].value_counts().head(15).reset_index()
        top_hs.columns = ['hashtag','count']
        fig_top = px.bar(
            top_hs,
            y='hashtag',
            x='count',
            orientation='h',
            text='count',
            color='count',
            color_continuous_scale='Blues'
        )
        fig_top.update_traces(textposition='outside')
        fig_top.update_layout(
            margin=dict(l=10,r=10,t=30,b=10),
            yaxis={'categoryorder':'total ascending'},
            showlegend=False,
            height=380
        )
        
        # Top hashtags by avg engagement
        perf = hs_df.groupby('hashtag', as_index=False).agg({
            'engagement': ['mean', 'count']
        })
        perf.columns = ['hashtag', 'engagement', 'usage_count']
        perf = perf[perf['usage_count'] >= 3].sort_values('engagement', ascending=False).head(15)
        
        if len(perf) > 0:
            fig_perf = px.bar(
                perf,
                y='hashtag',
                x='engagement',
                orientation='h',
                text='engagement',
                color='engagement',
                color_continuous_scale='Viridis',
                hover_data=['usage_count']
            )
            fig_perf.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig_perf.update_layout(
                margin=dict(l=10,r=10,t=30,b=10),
                yaxis={'categoryorder':'total ascending'},
                showlegend=False,
                height=380
            )
        else:
            fig_perf = px.bar(title='Not enough hashtag data (min 3 uses)')
    else:
        fig_top = px.bar(title='No hashtags in selection')
        fig_perf = px.bar(title='No hashtags in selection')
    
    # ====================================
    # 8. CAPTION LENGTH ANALYSIS
    # ====================================
    
    if len(f) > 0 and f['caption_length'].max() > 0:
        f_caption = f[f['caption_length'] > 0].copy()
        f_caption['caption_bin'] = pd.cut(
            f_caption['caption_length'],
            bins=[0, 50, 100, 200, 500, 2200],
            labels=['0-50', '51-100', '101-200', '201-500', '500+']
        )
        caption_stats = f_caption.groupby('caption_bin', observed=True)['engagement'].mean().reset_index()
        
        fig_caption = px.bar(
            caption_stats,
            x='caption_bin',
            y='engagement',
            text='engagement',
            color='engagement',
            color_continuous_scale='Teal'
        )
        fig_caption.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_caption.update_layout(
            margin=dict(l=10,r=10,t=30,b=10),
            xaxis_title='Caption Length (characters)',
            yaxis_title='Avg Engagement',
            showlegend=False,
            height=380
        )
    else:
        fig_caption = px.bar(title='No caption data')
    
    # ====================================
    # 9. HASHTAG COUNT ANALYSIS
    # ====================================
    
    if len(f) > 0 and f['hashtag_count'].max() > 0:
        hashtag_stats = f.groupby('hashtag_count')['engagement'].mean().reset_index()
        hashtag_stats = hashtag_stats[hashtag_stats['hashtag_count'] <= 30]
        
        if len(hashtag_stats) > 0:
            fig_hashtag_count = px.bar(
                hashtag_stats,
                x='hashtag_count',
                y='engagement',
                color='engagement',
                color_continuous_scale='Sunset'
            )
            fig_hashtag_count.update_layout(
                margin=dict(l=10,r=10,t=30,b=10),
                xaxis_title='Number of Hashtags',
                yaxis_title='Avg Engagement',
                showlegend=False,
                height=380
            )
        else:
            fig_hashtag_count = px.bar(title='No hashtag count data')
    else:
        fig_hashtag_count = px.bar(title='No hashtag data')
    
    # ====================================
    # 10 & 11. TOP AND BOTTOM POSTS
    # ====================================
    
    if len(f) >= 5:
        top_posts = f.nlargest(5, 'engagement')
        bottom_posts = f.nsmallest(5, 'engagement')
        
        def format_post_list(posts_df):
            items = []
            for idx, post in posts_df.iterrows():
                items.append(
                    html.Div([
                        html.Strong(f"{post['engagement']:.0f} engagement"),
                        html.Span(f" • {post['post_type']}", className='ms-2'),
                        html.Br(),
                        html.Small(post['caption_snippet'], className='text-muted'),
                        html.Br(),
                        html.A('View Post →', href=post['url'], target='_blank', 
                              className='btn btn-sm btn-outline-primary mt-1')
                    ], className='mb-3 pb-3 border-bottom')
                )
            return items
        
        top_posts_list = format_post_list(top_posts)
        bottom_posts_list = format_post_list(bottom_posts)
    else:
        top_posts_list = [html.P("Not enough data (need at least 5 posts)")]
        bottom_posts_list = [html.P("Not enough data (need at least 5 posts)")]
    
    # ====================================
    # 12. TABLE DATA
    # ====================================
    
    tbl = f.copy()
    if 'timestamp' in tbl.columns:
        tbl['date'] = tbl['timestamp'].dt.strftime('%Y-%m-%d %I:%M %p')
    else:
        tbl['date'] = ''
    
    tbl['link'] = tbl['url'].apply(lambda u: '[Open]('+u+')' if isinstance(u,str) and u.startswith('http') else '')
    tbl['engagement_score'] = tbl['engagement_score'].round(1)
    
    for col in ['likes', 'comments', 'views', 'engagement', 'hashtag_count']:
        tbl[col] = tbl[col].fillna(0).astype(int)
    
    cols = ['date','ownerUsername','post_type','likes','comments','views','engagement',
            'engagement_score','hashtag_count','caption_snippet','link']
    for c in cols:
        if c not in tbl.columns:
            tbl[c] = ''
    
    data = tbl[cols].sort_values('engagement', ascending=False).to_dict('records')
    
    # ====================================
    # RETURN ALL OUTPUTS
    # ====================================
    
    return (
        f"{total_posts:,}",
        posts_subtitle,
        f"{total_likes:,}",
        likes_subtitle,
        f"{total_comments:,}",
        comments_subtitle,
        f"{total_views:,}",
        views_subtitle,
        f"{avg_engagement:,.1f}",
        avg_eng_subtitle,
        f"{engagement_rate:.1f}",
        eng_rate_subtitle,
        f"{best_type} ({best_type_avg:.0f} avg)",
        f"{avg_hashtags:.1f}",
        f"{posts_per_week:.1f}",
        f"{consistency_score:.0f}%",
        fig_trend,
        fig_content,
        fig_dist,
        fig_corr,
        fig_heat,
        fig_top,
        fig_caption,
        fig_hashtag_count,
        fig_perf,
        top_posts_list,
        bottom_posts_list,
        data
    )

# ====================================
# CSV DOWNLOAD CALLBACK
# ====================================

@app.callback(
    Output('download-data','data'),
    [Input('btn-download','n_clicks'),
     State('owner-username-select','value'),
     State('date-range','start_date'),
     State('date-range','end_date'),
     State('post-type','value'),
     State('hashtag-select','value'),
     State('keyword-input','value')],
    prevent_initial_call=True
)
def download_csv(n, owner_usernames, start_date, end_date, post_types, hashtag_values, keyword):
    f = filter_df(df_raw, start_date, end_date, post_types, hashtag_values, keyword, owner_usernames)
    return dcc.send_data_frame(f.to_csv, 'instagram_filtered.csv', index=False)

# ====================================
# RUN APP
# ====================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("✅ Dashboard starting at http://localhost:8050")
    print("="*60)
    print("\n💡 Open http://localhost:8050 in your browser")
    print("⏹️  Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=8050)
