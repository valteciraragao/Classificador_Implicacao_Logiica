import streamlit as st
import re, joblib
import pandas as pd
from sympy import symbols, Implies, And, Or, Not
from sympy.logic.boolalg import truth_table

# ----------------------
# Tokenizer para ML
# ----------------------
def tokenizer(expr):
    return re.findall(r'->|\w+|[~&|()Â¬â†’]', expr)

# ----------------------
# Treinamento ML (uma vez)
# ----------------------
def treinar_ml():
    X = [
        "(p)->(p|q)",
        "(p&q)->p",
        "((p|q)&(~p))->q",
        "((p)&(p->q))->q",
        "((~q)&(p->q))->(~p)",
        "(p)->p",
        "(p&~p)->q",
    ]
    y = [
        "Tautologia",
        "Tautologia",
        "Tautologia",
        "Tautologia",
        "Tautologia",
        "ContingÃªncia",
        "Tautologia",
    ]
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    vec = CountVectorizer(tokenizer=tokenizer, token_pattern=None)
    clf = make_pipeline(vec, MultinomialNB())
    clf.fit(X, y)
    joblib.dump(clf, "ml_inferencia.joblib")
    return clf

# Carrega ou treina
try:
    ml = joblib.load("ml_inferencia_ampliado.joblib")
except:
    ml = treinar_ml()

# ----------------------
# ConversÃ£o NL â†’ simbÃ³lica
# ----------------------
def nl_to_expr(frase):
    text = frase.lower().strip().rstrip(".")
    m = re.match(r"(se|quando|caso)\s+(.*?)(?:,\s*ent[aÃ£]o\s+|\s+ent[aÃ£]o\s+)(.*)", text)
    if not m:
        return None, None, None, None
    P_txt = m.group(2).strip()
    Q_txt = m.group(3).strip()

    # Divide P e Q apenas por ' e ' e ' ou ' (nÃ£o toca no 'nÃ£o ')
    P_parts = re.split(r"\s+e\s+", P_txt)
    Q_parts = re.split(r"\s+ou\s+", Q_txt)

    # Lista de Ã¡tomos sem remover 'nÃ£o '
    atom = []
    for part in P_parts + Q_parts:
        a = part.strip()
        if a and a not in atom:
            atom.append(a)

    # Cria os sÃ­mbolos p, q, r...
    simb = symbols(' '.join(chr(112+i) for i in range(len(atom))))
    mp = {atom[i]: simb[i] for i in range(len(atom))}

    # Monta P_expr
    P_expr = None
    for part in P_parts:
        a = part.strip()
        if a.startswith("nÃ£o "):
            atom_expr = Not(mp[a])
        else:
            atom_expr = mp[a]
        P_expr = atom_expr if P_expr is None else And(P_expr, atom_expr)

    # Monta Q_expr
    Q_expr = None
    for part in Q_parts:
        a = part.strip()
        if a.startswith("nÃ£o "):
            atom_expr = Not(mp[a])
        else:
            atom_expr = mp[a]
        Q_expr = atom_expr if Q_expr is None else Or(Q_expr, atom_expr)

    expr = Implies(P_expr, Q_expr)
    sym_str = str(expr)
    return atom, expr, sym_str, Q_txt




# ----------------------
# Gera tabela e tipo
# ----------------------
def tabela_e_tipo(expr):
    vars_ = sorted(expr.free_symbols, key=lambda v: v.name)
    tbl = list(truth_table(expr, vars_))
    vals = [bool(r) for _, r in tbl]
    if all(vals):
        tipo = "Tautologia"
    elif not any(vals):
        tipo = "ContradiÃ§Ã£o"
    else:
        tipo = "ContingÃªncia"
    cols = [str(v) for v in vars_] + [str(expr)]
    data = []
    for row_vals, r in tbl:
        data.append([int(v) for v in row_vals] + [int(bool(r))])
    return cols, data, tipo

# ----------------------
# Inicia estado de histÃ³rico
# ----------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="InferÃªncias LÃ³gicas AvanÃ§adas", layout="wide")

# CSS tema escuro, header e inputs
st.markdown("""
    <style>
      body, .block-container, .main {
        background-color: #000000 !important;
        color: #EEEEEE !important;
      }
      .header-img {
        background-image: url('https://i.imgur.com/z4d4kWk.png');
        background-size: cover;
        height: 200px;
        border-radius: 10px;
        margin-bottom: 20px;
      }
      .stMarkdown p {
        font-size: 1.4em !important;
      }
      .stTextInput>div>input {
        background-color: #222222 !important;
        color: #EEEEEE !important;
        font-size: 1.2em !important;
        padding: 10px !important;
      }
      .stButton>button {
        background-color: #4a90e2 !important;
        color: #FFFFFF !important;
        font-size: 1.1em !important;
        padding: 8px 24px !important;
      }
      .stTable td, .stTable th {
        color: #EEEEEE !important;
      }
      hr { border-color: #444444 !important; }
    </style>
    <div class="header-img"></div>
""", unsafe_allow_html=True)

st.title("ðŸ”§ InferÃªncias LÃ³gicas com NL + ML")
st.markdown("<hr>", unsafe_allow_html=True)

frase = st.text_input("Digite sua condicional em portuguÃªs (Se P, entÃ£o Q):", "")

if frase:
    atom, expr, sym_str, Q_txt = nl_to_expr(frase)
    if not expr:
        st.error("Formato invÃ¡lido! Use â€˜Se P, entÃ£o Q.â€™")
    else:
        # AtÃ´micas
        st.subheader("ðŸ“Œ ProposiÃ§Ãµes AtÃ´micas")
        for i, a in enumerate(atom):
            cor = "#27ae60" if i % 2 == 0 else "#f39c12"
            st.markdown(f"<span style='color:{cor}'>**{chr(112+i)}:** {a}</span>", unsafe_allow_html=True)

        # SimbÃ³lica
        st.subheader("âš™ï¸ ExpressÃ£o SimbÃ³lica")
        st.markdown(f"<code style='color:#bbb'>{sym_str}</code>", unsafe_allow_html=True)

        # ML prevÃª
        pred = ml.predict([sym_str])[0]
        st.markdown(f"**[ML] Tipo previsto:** <span style='color:#e74c3c'>{pred}</span>", unsafe_allow_html=True)

        # Tabelaâ€‘verdade e contraâ€‘exemplo
        cols, data, real = tabela_e_tipo(expr)
        if not data:
            st.error("Erro ao gerar tabelaâ€‘verdade.")
        else:
            # Contraâ€‘exemplo
            counterexamples = [
                {**dict(zip(cols, row)), "idx": idx}
                for idx, row in enumerate(data) if row[-1] == 0
            ]
            if counterexamples:
                ce = counterexamples[0]
                vals = ", ".join(f"{col}={ce[col]}" for col in cols[:-1])
                st.warning(f"ðŸš¨ **Contraâ€‘exemplo (linha {ce['idx']}):** {vals} â‡’ **Resultado**=0")

            st.subheader("ðŸ§® Tabelaâ€‘Verdade")
            rows = [dict(zip(cols, row)) for row in data]
            st.table(rows)

        # Tipo exato
        cor_tipo = {"Tautologia":"#2ecc71","ContradiÃ§Ã£o":"#e74c3c","ContingÃªncia":"#f1c40f"}
        st.markdown(f"**[LÃ³gica Exata]**: <span style='color:{cor_tipo[real]}'>{real}</span>", unsafe_allow_html=True)

        # Feedback ML vs real
        if pred == real:
            st.success("âœ… ML e LÃ³gica concordam!")
            st.balloons()
        else:
            st.warning("âš ï¸ ML divergiu; confira a tabela acima.")

        # InterpretaÃ§Ã£o Natural com sujeito preenchido
        if Q_txt.lower().startswith("Ã© "):
            sujeito = atom[0].split(" ", 1)[0]
            Q_full = f"{sujeito} {Q_txt}"
        else:
            Q_full = Q_txt

        if real == "Tautologia":
            st.success("âœ¨ Esta Ã© uma implicaÃ§Ã£o lÃ³gica universal (tautologia)!")
            st.markdown(f"âœ… **InterpretaÃ§Ã£o Natural:**  \n> **{atom[0]}** implica que **{Q_full}**.")
        else:
            st.markdown(f"âŒ **InterpretaÃ§Ã£o Natural:**  \n> **{atom[0]}** *nÃ£o* implica que **{Q_full}** (tipo: *{real.lower()}*).")

# Atualiza histÃ³rico
        st.session_state.history.append({
            'Frase': frase,
            'SimbÃ³lica': sym_str,
            'ML': pred,
            'Exata': real
        })

        # Exibe histÃ³rico
        st.subheader("ðŸ“œ HistÃ³rico de AnÃ¡lises")
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist)

        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            "â¬‡ï¸ Baixar histÃ³rico CSV",
            data=csv,
            file_name="historico_inferencias.csv",
            mime="text/csv"
        )

# RodapÃ©
st.markdown("""
    <hr>
    <div style='display: flex; justify-content: space-between;
                font-size: 0.9em; color: #888; padding-top: 10px;'>
      <div>
        ðŸ‘¤ **Autores:** Valtecir AragÃ£o // Matheus Barbosa // Pedro Favato // Iago Xavier  
        ðŸŽ“ **Faculdade:** CEFET-RJ â€“ Sistemas de InformaÃ§Ã£o â€“ LÃ³gica Computacional
      </div>
      <div>
        ðŸ”— <a href='https://www.linkedin.com/in/valteciraragao' style='color:#4a90e2;'>LinkedIn</a>
      </div>
    </div>
""", unsafe_allow_html=True)
