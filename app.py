import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import quad

st.set_page_config(page_title="Romberg y crisis energética", page_icon="🌊", layout="wide")

AZUL, PASTEL, CELESTE, ACERO, MARINO = "#2F6FBF", "#8FB8E8", "#8ED3E6", "#5B7FA6", "#14284B"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&family=Quicksand:wght@500;600;700&display=swap');
.stApp {font-family:'Quicksand',sans-serif; color:#14284B;
  background:linear-gradient(180deg,#F2F8FF 0%,#E8F1FC 55%,#EAF7FB 100%);}
h1,h2,h3,h4 {font-family:'Fredoka',sans-serif !important; color:#1B3A6B;}
[data-testid="stSidebar"] {background:#E3EEFB; border-right:3px dashed #A9C6EA;}
.hero {background:linear-gradient(120deg,#BFD9F5,#9EC1EA 60%,#C9ECF5); border-radius:32px;
  padding:28px 34px; margin-bottom:18px; box-shadow:0 8px 0 #8DB0DB;}
.hero h1 {margin:0; font-size:2.3rem;}
.hero p {margin:6px 0 0 0; font-size:1.05rem;}
.chip {display:inline-block; background:#fff; border-radius:999px; padding:4px 14px; margin:8px 6px 0 0;
  border:2px solid #A9C6EA; font-weight:700; font-size:.85rem;}
.card {background:#fff; border-radius:24px; padding:16px 22px; border:2px solid #CFE0F5;
  box-shadow:0 6px 0 #CFE0F5; margin:10px 0;}
.step {font-family:'Fredoka',sans-serif; font-size:1.35rem; color:#1B3A6B; margin:26px 0 6px 0;}
.badge {display:inline-block; width:36px; height:36px; line-height:36px; text-align:center;
  background:#1F4E8C; color:#fff; border-radius:50%; margin-right:10px; box-shadow:0 3px 0 #143A6B;}
.stTabs [data-baseweb="tab"] {background:#fff; border-radius:999px; padding:8px 18px; margin-right:6px;
  border:2px solid #C5DAF2;}
.stTabs [aria-selected="true"] {background:#1F4E8C; border-color:#1F4E8C;}
.stTabs [aria-selected="true"] p {color:#fff;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
[data-testid="stMetric"] {background:#fff; border-radius:22px; padding:14px 18px;
  border:2px solid #CFE0F5; box-shadow:0 5px 0 #CFE0F5;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- núcleo numérico
def romberg(f, a, b, n):
    """Tabla de Romberg n x n (índices 0-based; R[k, j] = R_{k+1, j+1})."""
    R = np.full((n, n), np.nan)
    h = b - a
    R[0, 0] = h / 2 * (float(f(a)) + float(f(b)))
    for k in range(1, n):
        h /= 2
        pts = a + (2 * np.arange(1, 2 ** (k - 1) + 1) - 1) * h
        R[k, 0] = 0.5 * R[k - 1, 0] + h * float(np.sum(f(pts)))
        for j in range(1, k + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)
    return R


def step(num, title):
    st.markdown(f'<div class="step"><span class="badge">{num}</span>{title}</div>', unsafe_allow_html=True)


def card(html):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("### 🎛️ Panel de datos")
    modo = st.radio("Fuente de datos", ["Modelo con parámetros", "Tabla editable"])
    if modo == "Modelo con parámetros":
        objetivo = st.radio("¿Qué quieres calcular?", ["Energía no suministrada", "Energía total demandada"])
    else:
        objetivo = "Energía total demandada"
        N = st.selectbox("Puntos de la tabla (2^m + 1)", [9, 17, 33], index=1)
        st.caption("Con tabla se integra todo el día (0–24 h).")

    st.markdown("**Demanda de la ciudad (MW)**")
    base = st.number_input("Demanda base", 100.0, 3000.0, 520.0, 10.0)
    c1, c2 = st.columns(2)
    A1 = c1.number_input("Pico mañana", 0.0, 2000.0, 180.0, 10.0)
    A2 = c2.number_input("Pico noche", 0.0, 2000.0, 380.0, 10.0)
    mu1 = c1.number_input("Hora pico 1", 0.0, 24.0, 8.5, 0.5)
    mu2 = c2.number_input("Hora pico 2", 0.0, 24.0, 20.0, 0.5)
    s1 = c1.number_input("Ancho 1", 0.5, 8.0, 2.2, 0.1)
    s2 = c2.number_input("Ancho 2", 0.5, 8.0, 2.6, 0.1)

    st.markdown("**Oferta en crisis**")
    cap = st.number_input("Capacidad nominal (MW)", 100.0, 5000.0, 1000.0, 50.0)
    red = st.slider("Reducción por sequía/falla (%)", 0, 60, 30)
    G = cap * (1 - red / 100)

    if modo == "Modelo con parámetros":
        a, b = st.slider("Ventana de integración (h)", 0.0, 24.0, (18.0, 22.0), 0.5)
    else:
        a, b = 0.0, 24.0

    st.markdown("**Método**")
    n = st.slider("Niveles de Romberg (n)", 2, 8, 5) if modo == "Modelo con parámetros" else int(np.log2(N - 1)) + 1
    tol = st.select_slider("Tolerancia (MWh)", [1e-2, 1e-3, 1e-4, 1e-6, 1e-8], value=1e-4, format_func=lambda x: f"{x:.0e}")
    dec = st.slider("Decimales", 2, 10, 5)

    st.markdown("**Economía**")
    tarifa = st.number_input("Tarifa de generación (USD/MWh)", 1.0, 1000.0, 90.0, 5.0)
    voll = st.number_input("Costo de la energía no servida (USD/MWh)", 100.0, 20000.0, 3000.0, 100.0)
    kwh_hogar = st.number_input("Consumo diario de un hogar (kWh)", 1.0, 50.0, 8.0, 1.0)

if b - a <= 0:
    st.error("La ventana necesita b > a. Ajusta las horas en el panel.")
    st.stop()

fmt = lambda x: f"{x:.{dec}f}"
demanda_modelo = lambda t: base + A1 * np.exp(-(((t - mu1) / s1) ** 2)) + A2 * np.exp(-(((t - mu2) / s2) ** 2))
deficit = objetivo == "Energía no suministrada"

# ---------------------------------------------------------------- portada
st.markdown(
    """
<div class="hero">
<h1>⚡ Romberg frente a la crisis energética 🌊</h1>
<p>Un ejercicio de Ingeniería de Sistemas: ¿cuánta energía le va a faltar a la ciudad y cuánto cuesta ese apagón?</p>
<span class="chip">Integración numérica</span><span class="chip">Extrapolación de Richardson</span>
<span class="chip">Fiabilidad de sistemas</span><span class="chip">Datos editables</span>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["🌊 Contexto", "📝 Datos", "🧮 Paso a paso", "📊 Resultados", "🧠 Modelos aplicables"])

# ---------------------------------------------------------------- datos
with tabs[1]:
    st.markdown("### Datos del sistema")
    if modo == "Tabla editable":
        tt = np.linspace(0, 24, N)
        sig = hash((N, base, A1, A2, mu1, mu2, s1, s2))
        st.caption("Edita la demanda de cada hora. Si cambias los parámetros del panel, la tabla se reinicia.")
        df0 = pd.DataFrame({"Hora (h)": tt, "Demanda (MW)": np.round(demanda_modelo(tt), 1)})
        ed = st.data_editor(df0, key=f"ed_{sig}", disabled=["Hora (h)"], hide_index=True)
        yv = ed["Demanda (MW)"].to_numpy(float)
        dem = lambda t: np.interp(t, tt, yv)
    else:
        dem = demanda_modelo
        st.caption("La demanda es la suma de una base y dos picos (mañana y noche). Cambia los valores en el panel de la izquierda.")

    f = (lambda t: dem(t) - G) if deficit else dem

    tf = np.linspace(0, 24, 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tf, y=np.full_like(tf, G), name="Oferta disponible", line=dict(color=PASTEL, dash="dash", width=3)))
    xw = np.linspace(a, b, 200)
    if deficit:
        fig.add_trace(go.Scatter(x=xw, y=np.full_like(xw, G), showlegend=False, line=dict(width=0)))
        fig.add_trace(go.Scatter(x=xw, y=dem(xw), fill="tonexty", fillcolor="rgba(142,211,230,.55)", name="Energía no suministrada", line=dict(width=0)))
    else:
        fig.add_trace(go.Scatter(x=xw, y=dem(xw), fill="tozeroy", fillcolor="rgba(142,211,230,.45)", name="Energía a integrar", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=tf, y=dem(tf), name="Demanda D(t)", line=dict(color=AZUL, width=4)))
    fig.update_layout(height=380, xaxis_title="Hora del día", yaxis_title="MW", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#fff", legend=dict(orientation="h", y=1.12), font=dict(color=MARINO))
    st.plotly_chart(fig)

    if deficit and modo == "Modelo con parámetros" and np.any(dem(xw) < G):
        st.warning("En parte de la ventana la oferta supera a la demanda: esa zona restaría energía. Ajusta la ventana a las horas con déficit.")
    st.metric("Oferta disponible en crisis", f"{G:,.0f} MW", f"-{red}% de la capacidad nominal")

# ---------------------------------------------------------------- cálculo
R = romberg(f, a, b, n)
exact = None
if modo == "Modelo con parámetros":
    exact = quad(lambda t: float(f(t)), a, b, epsabs=1e-13, epsrel=1e-13)[0]
conv = next((k + 1 for k in range(1, n) if abs(R[k, k] - R[k - 1, k - 1]) < tol), None)
best = R[conv - 1, conv - 1] if conv else R[n - 1, n - 1]
lvl_best = conv or n
evals = 2 ** (lvl_best - 1) + 1

# ---------------------------------------------------------------- contexto
with tabs[0]:
    st.markdown("### El problema")
    card(
        "Una sequía redujo la capacidad de las centrales hidroeléctricas. En la <b>hora punta de la noche</b> la demanda "
        "supera a la oferta y el operador del sistema tiene que cortar carga. Como ingeniero/a de sistemas necesitas "
        "cuantificar la <b>energía no suministrada (ENS)</b> y su costo para decidir si conviene alquilar generación de emergencia."
    )
    st.markdown("La energía es el área bajo la curva de potencia, es decir, una integral:")
    st.latex(r"\text{ENS}=\int_a^b \big(D(t)-G\big)\,dt \qquad [\text{MW}\cdot\text{h}=\text{MWh}]")
    card(
        "💡 <b>¿Por qué Romberg?</b> La curva de demanda es suave, justo el caso donde Romberg brilla: "
        "con pocas evaluaciones de la función logra una precisión que el trapecio simple tardaría miles de pasos en alcanzar. "
        "Si cada evaluación fuese una simulación costosa del sistema eléctrico, eso ahorra mucho tiempo."
    )
    st.info("Ve a **Datos** para editar el escenario y a **Paso a paso** para resolverlo.")

# ---------------------------------------------------------------- paso a paso
with tabs[2]:
    what = "ENS" if deficit else "E"
    step(1, "Planteamos la integral")
    if deficit:
        st.latex(rf"\text{{ENS}}=\int_{{{a:g}}}^{{{b:g}}}\big(D(t)-{G:g}\big)dt")
    else:
        st.latex(rf"E=\int_{{{a:g}}}^{{{b:g}}}D(t)\,dt")
    if modo == "Modelo con parámetros":
        st.latex(rf"D(t)={base:g}+{A1:g}\,e^{{-((t-{mu1:g})/{s1:g})^2}}+{A2:g}\,e^{{-((t-{mu2:g})/{s2:g})^2}}")
    else:
        st.write("D(t) viene de tu tabla; los valores intermedios se toman de ella.")

    step(2, "Nivel 1: trapecio con un solo intervalo")
    h1 = b - a
    st.latex(rf"R_{{1,1}}=\frac{{h}}{{2}}\,[f(a)+f(b)]=\frac{{{h1:g}}}{{2}}\,[{float(f(a)):.4f}+{float(f(b)):.4f}]={fmt(R[0, 0])}")

    def fig_trap(k):
        m = 2 ** (k - 1)
        xs = np.linspace(a, b, m + 1)
        xx = np.linspace(a, b, 300)
        fg = go.Figure()
        fg.add_trace(go.Scatter(x=xs, y=f(xs), fill="tozeroy", fillcolor="rgba(143,184,232,.35)", mode="lines+markers",
                                line=dict(color=PASTEL, width=2), marker=dict(size=8, color=CELESTE, line=dict(width=1, color=MARINO)), name=f"{m} trapecio(s)"))
        fg.add_trace(go.Scatter(x=xx, y=f(xx), line=dict(color=AZUL, width=3), name="f(t)"))
        fg.update_layout(height=300, margin=dict(t=30, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
                         title=f"Nivel {k}: {m} trapecio(s), h = {(b - a) / m:g}", font=dict(color=MARINO), legend=dict(orientation="h", y=-0.2))
        return fg

    st.plotly_chart(fig_trap(1))

    step(3, "Nivel 2: partimos el intervalo a la mitad")
    h2 = (b - a) / 2
    xm = a + h2
    st.latex(rf"R_{{2,1}}=\tfrac12 R_{{1,1}}+h\,f(a+h)=\tfrac12({fmt(R[0, 0])})+{h2:g}\cdot({float(f(xm)):.4f})={fmt(R[1, 0])}")
    st.caption("Solo se evalúa el punto nuevo; los demás se reutilizan.")
    st.plotly_chart(fig_trap(2))

    step(4, "Extrapolación de Richardson: eliminamos el error")
    st.markdown("El trapecio comete un error del orden de h². Combinando dos aproximaciones con pasos h y h/2 se cancela ese término:")
    st.latex(r"R_{k,j}=R_{k,j-1}+\frac{R_{k,j-1}-R_{k-1,j-1}}{4^{\,j-1}-1}")
    st.latex(rf"R_{{2,2}}={fmt(R[1, 0])}+\frac{{{fmt(R[1, 0])}-{fmt(R[0, 0])}}}{{3}}={fmt(R[1, 1])}")

    step(5, "Explora cualquier nivel")
    kk = st.radio("Nivel k", list(range(2, n + 1)), horizontal=True, index=min(1, n - 2))
    K = kk - 1
    hk = (b - a) / 2**K
    nuevos = a + (2 * np.arange(1, 2 ** (kk - 2) + 1) - 1) * hk
    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown(f"**Puntos nuevos** (h = {hk:g}) — {len(nuevos)} evaluación(es):")
        st.dataframe(pd.DataFrame({"t (h)": nuevos[:12], "f(t)": f(nuevos[:12])}), hide_index=True)
        if len(nuevos) > 12:
            st.caption(f"Se muestran 12 de {len(nuevos)}.")
        st.latex(rf"R_{{{kk},1}}=\tfrac12\,({fmt(R[K - 1, 0])})+{hk:g}\cdot({float(np.sum(f(nuevos))):.4f})={fmt(R[K, 0])}")
        for j in range(2, kk + 1):
            st.latex(rf"R_{{{kk},{j}}}={fmt(R[K, j - 2])}+\frac{{{fmt(R[K, j - 2])}-{fmt(R[K - 1, j - 2])}}}{{{4 ** (j - 1) - 1}}}={fmt(R[K, j - 1])}")
    with cB:
        st.plotly_chart(fig_trap(kk))

    step(6, "Tabla de Romberg completa")
    cols = [f"O(h^{2 * (j + 1)})" for j in range(n)]
    tab = pd.DataFrame(
        [[fmt(v) if not np.isnan(v) else "" for v in row] for row in R],
        columns=cols,
        index=[f"k={k + 1} (h={(b - a) / 2**k:g})" for k in range(n)],
    )
    diag = lambda d: pd.DataFrame(
        [[f"background-color:#CFE0F5;font-weight:700" if i == j else "" for j in range(d.shape[1])] for i in range(d.shape[0])],
        index=d.index, columns=d.columns)
    st.dataframe(tab.style.apply(diag, axis=None))
    st.caption("La diagonal (azul) es la mejor aproximación de cada nivel.")

    step(7, "Criterio de parada")
    st.latex(rf"|R_{{k,k}}-R_{{k-1,k-1}}|<{tol:.0e}\ \text{{MWh}}")
    dd = pd.DataFrame({"k": range(2, n + 1), "|R(k,k) − R(k−1,k−1)|": [abs(R[k, k] - R[k - 1, k - 1]) for k in range(1, n)]})
    dd["¿Cumple?"] = np.where(dd.iloc[:, 1] < tol, "✅", "—")
    st.dataframe(dd, hide_index=True)
    if conv:
        st.success(f"Converge en el nivel {conv}: {fmt(best)} MWh con solo {evals} evaluaciones de la función.")
    else:
        st.warning("No alcanzó la tolerancia: sube el número de niveles en el panel.")

# ---------------------------------------------------------------- resultados
with tabs[3]:
    st.markdown("### Resultados e interpretación")
    m1, m2, m3 = st.columns(3)
    m1.metric("Energía calculada (Romberg)", f"{best:,.{min(dec, 4)}f} MWh")
    m2.metric("Evaluaciones de f", f"{evals}")
    m3.metric("Error estimado", f"{abs(R[lvl_best - 1, lvl_best - 1] - R[lvl_best - 2, lvl_best - 2]):.1e} MWh")
    m4, m5, m6 = st.columns(3)
    if deficit:
        m4.metric("Costo social del apagón", f"USD {best * voll:,.0f}")
        m5.metric("Costo si se genera a tarifa", f"USD {best * tarifa:,.0f}")
        m6.metric("Equivale a hogares sin luz un día", f"{best * 1000 / kwh_hogar:,.0f}")
    else:
        m4.metric("Costo de generación", f"USD {best * tarifa:,.0f}")
        m5.metric("Energía media", f"{best / (b - a):,.1f} MW")
        m6.metric("Hogares abastecidos (día)", f"{best * 1000 / kwh_hogar:,.0f}")
    if exact is not None:
        st.caption(f"Valor de referencia (scipy.quad): {exact:.8f} MWh · error real de Romberg: {abs(best - exact):.2e}")

    ks = np.arange(1, n + 1)
    fg = go.Figure()
    est = [np.nan] + [max(abs(R[k, k] - R[k - 1, k - 1]), 1e-16) for k in range(1, n)]
    fg.add_trace(go.Scatter(x=ks, y=est, name="Error estimado (diagonal)", mode="lines+markers", line=dict(color=PASTEL, width=3)))
    if exact is not None:
        fg.add_trace(go.Scatter(x=ks, y=np.maximum(np.abs(R[:, 0] - exact), 1e-16), name="Error del trapecio", mode="lines+markers", line=dict(color=ACERO, width=3)))
        fg.add_trace(go.Scatter(x=ks, y=np.maximum(np.abs(np.diag(R) - exact), 1e-16), name="Error de Romberg", mode="lines+markers", line=dict(color=AZUL, width=3)))
    fg.update_layout(height=360, yaxis_type="log", xaxis_title="Nivel k", yaxis_title="Error (MWh, escala log)",
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", font=dict(color=MARINO), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fg)
    card("💡 <b>Lectura:</b> el error del trapecio baja lento, mientras que el de Romberg cae varios órdenes de magnitud en pocos niveles. "
         "Esa es la ganancia de la extrapolación.")
    if deficit:
        st.markdown(f"**Decisión de ingeniería:** el apagón costaría unos **USD {best * voll:,.0f}** frente a **USD {best * tarifa:,.0f}** "
                    f"de generar esa energía a tarifa normal. Alquilar generación de emergencia a un precio menor que el costo social es rentable.")

# ---------------------------------------------------------------- modelos
with tabs[4]:
    st.markdown("### ¿En qué modelos matemáticos se puede aplicar Romberg?")
    st.caption("Siempre que el modelo tenga una integral definida de una función suave y evaluarla sea costoso.")
    modelos = [
        ("🔋", "Balance de energía y fiabilidad del suministro", r"\text{EENS}=\int_0^T \max\big(D(t)-G(t),0\big)\,dt",
         "Energía esperada no servida; se aplica por tramos suaves del déficit."),
        ("🛠️", "Fiabilidad: tasa de falla y MTTF", r"R(t)=e^{-\int_0^t\lambda(s)ds},\quad \text{MTTF}=\int_0^\infty R(t)\,dt",
         "Vida útil de turbinas y transformadores con tasa de falla λ(t) no constante."),
        ("🌊", "Dinámica de sistemas (niveles y flujos)", r"\text{Nivel}(T)=\text{Nivel}(0)+\int_0^T\big(\text{entradas}-\text{salidas}\big)dt",
         "Embalses, inventario de combustible o reservas de energía en un diagrama de Forrester."),
        ("🎛️", "Control: índices de desempeño", r"\text{IAE}=\int_0^T|e(t)|\,dt,\quad \text{ISE}=\int_0^T e(t)^2\,dt",
         "Evaluar qué tan bien un regulador de frecuencia sigue la consigna."),
        ("💰", "Valor presente en tiempo continuo", r"VPN=\int_0^T C(t)\,e^{-rt}\,dt",
         "Costo del ciclo de vida de una inversión en renovables o almacenamiento."),
        ("🎲", "Probabilidad y valor esperado", r"P(a\le X\le b)=\int_a^b f_X(x)dx,\quad E[X]=\int x f_X(x)dx",
         "Demanda incierta, vientos (Weibull) o irradiancia; simulación estocástica."),
    ]
    c1, c2 = st.columns(2)
    for i, (ic, nom, ltx, txt) in enumerate(modelos):
        with (c1 if i % 2 == 0 else c2):
            card(f"<b>{ic} {nom}</b><br>{txt}")
            st.latex(ltx)
    card("⚠️ <b>Cuándo NO usarlo:</b> funciones con saltos, esquinas o singularidades (por ejemplo un max(D−G, 0) sobre todo el día) "
         "o datos muy ruidosos. Ahí conviene dividir el intervalo en tramos suaves o usar integración adaptativa.")
