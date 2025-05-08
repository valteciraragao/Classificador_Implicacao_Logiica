import streamlit as st import re import joblib import pandas as pd from sympy import symbols, Implies, And, Or, Not from sympy.logic.boolalg import truth_table

----------------------

Tokenizer para ML

----------------------

def tokenizer(expr): return re.findall(r'->|→|\w+|[~&|()¬]', expr)

----------------------

Treinamento ML (uma vez)

----------------------

def treinar_ml(): X = [ "(p)->(p|q)", "(p&q)->p", "((p|q)&(~p))->q", "((p)&(p->q))->q", "((~q)&(p->q))->(~p)", "(p)->p", "(p&~p)->q", ] y = ["Tautologia"]*5 + ["Contingência", "Tautologia"]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    CountVectorizer(tokenizer=tokenizer, token_pattern=None),
    MultinomialNB()
)
clf.fit(X, y)
joblib.dump(clf, "ml_inferencia.joblib")
return clf

Carrega ou treina

try: ml = joblib.load("ml_inferencia.joblib") except FileNotFoundError: ml = treinar_ml()

----------------------

Conversão NL → simbólica

----------------------

def nl_to_expr(frase): # Normaliza texto text = frase.lower().strip() text = text.replace("entao", "então").rstrip('.') # Regex para P e Q m = re.match(r"^(se|quando|caso)\s+(.?)(?:,\sent[aã]o\s+|\s+ent[aã]o\s+)(.*)$", text) if not m: return None, None, None, None

P_txt = m.group(2).strip()
Q_txt = m.group(3).strip()

# Divide conjunções e disjunções simples
P_parts = re.split(r"\s+e\s+", P_txt)
Q_parts = re.split(r"\s+ou\s+", Q_txt)

# Coleta átomos
atom = []
for part in P_parts + Q_parts:
    a = part.strip()
    if a and a not in atom:
        atom.append(a)

# Gera símbolos p, q, r...
simb = symbols(' '.join(chr(112 + i) for i in range(len(atom))))

# Verifica tamanhos iguais
if len(atom) != len(simb):
    raise ValueError(f"Erro ao mapear símbolos: atom={atom}, simb={simb}")

mp = {atom[i]: simb[i] for i in range(len(atom))}

# Monta expressão P
P_expr = None
for part in P_parts:
    a = part.strip()
    if a.startswith("não "):
        name = a[5:].strip()
        atom_expr = Not(mp.get(name, mp.get(a)))
    else:
        atom_expr = mp[a]
    P_expr = atom_expr if P_expr is None else And(P_expr, atom_expr)

# Monta expressão Q
Q_expr = None
for part in Q_parts:
    a = part.strip()
    if a.startswith("não "):
        name = a[5:].strip()
        atom_expr = Not(mp.get(name, mp.get(a)))
    else:
        atom_expr = mp[a]
    Q_expr = atom_expr if Q_expr is None else Or(Q_expr, atom_expr)

expr = Implies(P_expr, Q_expr)
sym_str = str(expr)
return atom, expr, sym_str, Q_txt

----------------------

Gera tabela e tipo

----------------------

def tabela_e_tipo(expr): vars_ = sorted(expr.free_symbols, key=lambda v: v.name) tbl = list(truth_table(expr, vars_)) vals = [bool(row[1]) for row in tbl]

if all(vals):
    tipo = "Tautologia"
elif not any(vals):
    tipo = "Contradição"
else:
    tipo = "Contingência"

cols = [str(v) for v in vars_] + [str(expr)]
data = []
for row in tbl:
    vals_row, result = row[0], row[1]
    data.append([int(v) for v in vals_row] + [int(bool(result))])

return cols, data, tipo

----------------------

Inicia estado de histórico

----------------------

if 'history' not in st.session_state: st.session_state.history = []

----------------------

Streamlit UI

----------------------

st.set_page_config(page_title="Inferências Lógicas Avançadas", layout="wide")

st.title("🔧 Inferências Lógicas com NL + ML")

frase = st.text_input( "Digite sua condicional em português (ex: Se P, então Q):", "" )

if frase: atom, expr, sym_str, Q_txt = nl_to_expr(frase) if expr is None: st.error("⚠️ Formato inválido! Use 'Se P, então Q.'") else: # Átomos st.subheader("📌 Proposições Atômicas") for i, a in enumerate(atom): cor = "#27ae60" if i % 2 == 0 else "#f39c12" st.markdown(f"{chr(112+i)}: {a}")

# Expressão simbólica
    st.subheader("⚙️ Expressão Simbólica")
    st.code(sym_str)

    # ML prevê
    pred = ml.predict([sym_str])[0]
    st.markdown(f"**[ML] Tipo previsto:** **{pred}**")

    # Tabela-verdade
    cols, data, real = tabela_e_tipo(expr)
    # Contra-exemplo
    counter = [(idx, row) for idx, row in enumerate(data) if row[-1] == 0]
    if counter:
        idx, row = counter[0]
        vals = ", ".join(f"{cols[i]}={row[i]}" for i in range(len(cols)-1))
        st.warning(f"🚨 Contra-exemplo (linha {idx}): {vals} ⇒ Resultado=0")

    st.subheader("🧮 Tabela‑Verdade")
    df = pd.DataFrame(data, columns=cols)
    st.table(df)

    # Tipo exato
    cor_tipo = {"Tautologia": "#2ecc71", "Contradição": "#e74c3c", "Contingência": "#f1c40f"}
    st.markdown(f"**[Lógica Exata]:** **{real}**")

    # Feedback
    if pred == real:
        st.success("✅ ML e Lógica concordam!")
        st.balloons()
    else:
        st.warning("⚠️ ML divergiu; confira acima.")

    # Interpretação Natural
    P_txt = atom[0]
    if real == "Tautologia":
        st.markdown(f"✅ **Interpretação Natural:** {P_txt} implica que {Q_txt}.")
    else:
        st.markdown(f"❌ **Interpretação Natural:** {P_txt} NÃO implica que {Q_txt}.")

    # Histórico
    st.session_state.history.append({
        'Frase': frase,
        'Simbólica': sym_str,
        'ML': pred,
        'Exata': real
    })

    st.subheader("📜 Histórico de Análises")
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist)

    csv = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Baixar histórico CSV",
        data=csv,
        file_name="historico_inferencias.csv",
        mime="text/csv"
    )

Rodapé

st.markdown( """ 👤 Autores: Valtecir Aragão // Matheus Barbosa // Pedro Favato // Iago Xavier 🎓 Faculdade: CEFET-RJ – Sistemas de Informação – Lógica Computacional

🔗 [LinkedIn](https://www.linkedin.com/in/valteciraragao)
"""

)

