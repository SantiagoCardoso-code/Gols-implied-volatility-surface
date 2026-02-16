import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime
import pytz
from scipy.interpolate import Rbf # <--- THE SECRET SAUCE FOR SMOOTHING

# --- 1. SANDBOX CACHE (Prevents Disk Error) ---
cache_dir = os.path.join(tempfile.gettempdir(), f"yf_quant_v24_{datetime.now().strftime('%Y%m%d%H%M')}")
if not os.path.exists(cache_dir): os.makedirs(cache_dir)
yf.set_tz_cache_location(cache_dir)

app = dash.Dash(__name__)
portugal_tz = pytz.timezone('Europe/Lisbon')

# --- STYLE CONSTANTS ---
COLORS = {
    'bg': '#0e1117',        # Dark Blue-Grey (VS Code style)
    'card': '#161b22',      # Lighter panel
    'text': '#c9d1d9',      # Soft white
    'gold': '#d29922',      # Muted Gold
    'green': '#2ea043',     # Git Green
    'red': '#f85149',       # Git Red
    'grid': '#30363d'       # Grid lines
}

app.layout = html.Div([
    dcc.Store(id='camera-store', data={'eye': {'x': 1.8, 'y': 1.8, 'z': 0.8}}), # Remembers View

    # --- TOP BAR ---
    html.Div([
        html.Div([
            html.H1("VOLATILITY SURFACE // PRO", style={'color': COLORS['text'], 'fontSize': '24px', 'fontWeight': 'bold', 'margin': '0', 'display': 'inline-block'}),
            html.Span(" | LIVE FEED", style={'color': COLORS['green'], 'fontSize': '14px', 'marginLeft': '10px', 'fontWeight': 'bold'})
        ], style={'display': 'inline-block'}),
        
        html.Div(id='live-clock', style={'float': 'right', 'color': COLORS['text'], 'fontFamily': 'monospace', 'fontSize': '18px'}),
    ], style={'padding': '15px 25px', 'backgroundColor': COLORS['card'], 'borderBottom': f'1px solid {COLORS['grid']}'}),

    # --- CONTROLS STRIP ---
    html.Div([
        # Manual Spot
        html.Label("SPOT PX:", style={'color': COLORS['gold'], 'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Input(id='manual-spot', type='number', placeholder='AUTO', style={'backgroundColor': COLORS['bg'], 'color': 'white', 'border': f'1px solid {COLORS['grid']}', 'padding': '5px', 'borderRadius': '4px', 'width': '80px'}),
        
        # View Lock
        html.Button('🔒 LOCK CAMERA', id='lock-btn', n_clicks=0, style={'marginLeft': '20px', 'backgroundColor': COLORS['grid'], 'color': 'white', 'border': 'none', 'padding': '6px 12px', 'borderRadius': '4px', 'cursor': 'pointer'}),
        html.Span(id='lock-status', children="UNLOCKED", style={'marginLeft': '10px', 'fontSize': '12px', 'color': COLORS['text']}),

        # Refresh
        html.Button('⚡ UPDATE', id='update-btn', n_clicks=0, style={'float': 'right', 'backgroundColor': COLORS['green'], 'color': 'white', 'border': 'none', 'padding': '6px 15px', 'borderRadius': '4px', 'fontWeight': 'bold', 'cursor': 'pointer'})
    ], style={'padding': '10px 25px', 'backgroundColor': COLORS['bg'], 'borderBottom': f'1px solid {COLORS['grid']}'}),

    # --- MAIN VISUALIZATION ---
    html.Div([
        dcc.Graph(id='vol-surface-3d', style={'height': '75vh'}, config={'displayModeBar': False})
    ], style={'backgroundColor': 'black'}),

    # --- ANALYTICS FOOTER ---
    html.Div([
        html.Div(id='quant-metrics', style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '10px', 'color': COLORS['text'], 'fontFamily': 'monospace', 'fontSize': '16px'})
    ], style={'padding': '20px', 'backgroundColor': COLORS['card'], 'borderTop': f'1px solid {COLORS['grid']}'}),

    dcc.Interval(id='fast-tick', interval=1000, n_intervals=0)

], style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'})


# --- CLOCK ---
@app.callback(Output('live-clock', 'children'), Input('fast-tick', 'n_intervals'))
def update_time(n):
    return datetime.now(portugal_tz).strftime("%H:%M:%S")

# --- LOCK LOGIC ---
@app.callback(
    [Output('lock-status', 'children'), Output('lock-status', 'style')],
    Input('lock-btn', 'n_clicks')
)
def toggle_lock(n):
    if n % 2 == 1:
        return "LOCKED", {'marginLeft': '10px', 'fontSize': '12px', 'color': COLORS['red'], 'fontWeight': 'bold'}
    return "UNLOCKED", {'marginLeft': '10px', 'fontSize': '12px', 'color': COLORS['text']}

# --- QUANT CORE ---
@app.callback(
    [Output('vol-surface-3d', 'figure'), Output('quant-metrics', 'children')],
    [Input('update-btn', 'n_clicks')],
    [State('manual-spot', 'value'), State('lock-btn', 'n_clicks'), State('vol-surface-3d', 'relayoutData')]
)
def update_surface(n, manual_price, lock_clicks, relayout):
    try:
        # 1. DATA INGESTION
        ticker = yf.Ticker("GLD")
        if manual_price:
            spot = float(manual_price)
            spot_source = "MANUAL"
        else:
            hist = ticker.history(period="1d")
            spot = hist['Close'].iloc[-1] * 10.885
            spot_source = "FEED"

        # 2. OPTION CHAIN SCANNER
        raw_points = []
        for exp in ticker.options[1:5]:
            chain = ticker.option_chain(exp)
            days = (datetime.strptime(exp, '%Y-%m-%d').date() - datetime.now().date()).days
            
            # Filter Logic: +/- 15% from Spot to focus the surface
            calls = chain.calls
            mask = (calls['strike'] * 10.885 > spot * 0.85) & (calls['strike'] * 10.885 < spot * 1.15)
            df_opt = calls[mask]
            
            for _, row in df_opt.iterrows():
                raw_points.append({
                    'x': days,
                    'y': row['strike'] * 10.885,
                    'z': row['impliedVolatility']
                })

        df = pd.DataFrame(raw_points)

        # 3. RBF INTERPOLATION (The "Smooth" Math)
        # We create a dense grid to map the "Smooth" surface onto
        ti_x = np.linspace(df['x'].min(), df['x'].max(), 30)
        ti_y = np.linspace(df['y'].min(), df['y'].max(), 30)
        XI, YI = np.meshgrid(ti_x, ti_y)
        
        # Radial Basis Function Interpolator (Linear type works best for Vol Surfaces)
        rbf = Rbf(df['x'], df['y'], df['z'], function='linear')
        ZI = rbf(XI, YI)

        # 4. REGIME STATS
        # Calculate Skew at nearest expiration
        near_term = df[df['x'] == df['x'].min()]
        if not near_term.empty:
            atm_vol = near_term.iloc[(near_term['y'] - spot).abs().argsort()[:1]]['z'].values[0]
            otm_target = spot + 300
            otm_vol_series = near_term.iloc[(near_term['y'] - otm_target).abs().argsort()[:1]]['z']
            otm_vol = otm_vol_series.values[0] if not otm_vol_series.empty else atm_vol
            skew = otm_vol - atm_vol
        else:
            skew = 0.0

        # 5. CAMERA HANDLING
        camera = None
        if lock_clicks % 2 == 1 and relayout and 'scene.camera' in relayout:
            camera = relayout['scene.camera']

        # 6. PLOTTING
        fig = go.Figure(data=[
            # The Smooth Surface
            go.Surface(x=XI, y=YI, z=ZI, colorscale='Viridis', opacity=0.9, showscale=False),
            # The Real Data Points (Dots) - Optional, shows accuracy
            go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers', marker=dict(size=2, color='white', opacity=0.5))
        ])

        fig.update_layout(
            scene=dict(
                xaxis_title='DTE',
                yaxis_title='STRIKE',
                zaxis_title='IV',
                xaxis=dict(backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid'], title_font=dict(color=COLORS['text'])),
                yaxis=dict(backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid'], title_font=dict(color=COLORS['text'])),
                zaxis=dict(backgroundcolor=COLORS['bg'], gridcolor=COLORS['grid'], title_font=dict(color=COLORS['text'])),
                camera=camera
            ),
            paper_bgcolor='black',
            margin=dict(l=0, r=0, b=0, t=0),
            uirevision='locked' if lock_clicks % 2 == 1 else 'unlocked'
        )

        # 7. METRICS OUTPUT
        regime_color = COLORS['red'] if skew > 0.04 else COLORS['green'] if skew < -0.01 else COLORS['gold']
        metrics = [
            html.Div([html.Span("SPOT PRICE: "), html.B(f"${spot:.2f}")]),
            html.Div([html.Span("ATM VOL: "), html.B(f"{atm_vol:.2%}")]),
            html.Div([html.Span("SKEW (+300): "), html.B(f"{skew:.4f}", style={'color': regime_color})]),
            html.Div([html.Span("REGIME: "), html.B("BEARISH" if skew > 0.02 else "BULLISH" if skew < 0 else "NEUTRAL", style={'color': regime_color})])
        ]

        return fig, metrics

    except Exception as e:
        return go.Figure(), [html.Div(f"DATA ERROR: {str(e)}", style={'color': 'red'})]

if __name__ == '__main__':
    app.run(debug=False, port=8050)