import streamlit as st
from sympy import symbols, Implies, Not, And, Or
from sympy.logic.boolalg import truth_table
import re
import pandas as pd

st.set_page_config(page_title="LogicAI - Analisador Lógico", layout="centered")

st.title("LogicAI: Interprete e Classifique Implicações Lógicas")
st.markdown("Digite uma frase condicional em linguagem natural, como:\n\n**Se está chovendo e é terça-feira, então levo guarda-chuva.**")

# Entrada da frase
frase = st.text_area("Frase lógica em linguagem natural", height=100)

# Função para extrair proposições e montar expressão lógica
def nl_to_expr(frase):
    text = frase.lower().strip().rstrip(".")

    # Regex robusto para separar a implicação
    m = re.match(r"(se|quando|caso)\s+(.*?)(?:,\s*ent[aã]o\s+|\s+ent[aã]o\s+)(.*)", text)
    if not m:
        return [], None, None, None

    P_txt = m.group(2).strip()
    Q_txt = m.group(3).strip()

    # Dividir proposições
    P_parts = re.split(r"\s+e\s+", P_txt)
    Q_parts = re.split(r"\s+ou\s+", Q_txt)

    atom = []
    for part in P_parts + Q_parts:
        a = part.strip()
        if a and a not in atom:
            atom.append(a)

    simb = symbols(' '.join(chr(112+i) for i in range(len(atom))))  # p, q, r...
    mp = {atom[i]: simb[i] for i in range(len(atom))}

    # Construir P (com E)
    P_expr = None
    for part in P_parts:
        a = part.strip()
        atom_expr = Not(mp[a[4:].strip()]) if a.startswith("não ") else mp[a]
        P_expr = atom_expr if P_expr is None else And(P_expr, atom_expr)

    # Construir Q (com OU)
    Q_expr = None
    for part in Q_parts:
        a = part.strip()
        atom_expr = Not(mp[a[4:].strip()]) if a.startswith("não ") else mp[a]
        Q_expr = atom_expr if Q_expr is None else Or(Q_expr, atom_expr)

    expr = Implies(P_expr, Q_expr)
    sym_str = str(expr)
    return atom, expr, sym_str, Q_txt

# Função para gerar a tabela verdade
def gerar_tabela_verdade(expr, atom):
    simb = symbols(' '.join(chr(112+i) for i in range(len(atom))))
    tabela = list(truth_table(expr, simb))
    data = []
    for linha in tabela:
        valores, resultado = linha
        linha_dict = {str(simb[i]): valores[i] for i in range(len(simb))}
        linha_dict["Resultado"] = resultado
        data.append(linha_dict)
    return pd.DataFrame(data)

# Classificador da implicação
def classificar_implicacao(tabela):
    col_resultado = tabela["Resultado"].tolist()
    if all(col_resultado):
        return "Tautologia (sempre verdadeira)"
    elif not any(col_resultado):
        return "Contradição (sempre falsa)"
    else:
        return "Contingência (às vezes verdadeira)"

# Execução ao clicar
if frase:
    atom, expr, sym_str, Q_txt = nl_to_expr(frase)

    if not atom or expr is None:
        st.error("⚠️ Frase mal formatada. Use o modelo: 'Se P, então Q'")
    else:
        st.subheader("Expressão lógica gerada:")
        st.latex(sym_str)

        df = gerar_tabela_verdade(expr, atom)
        st.subheader("Tabela Verdade")
        st.dataframe(df, use_container_width=True)

        tipo = classificar_implicacao(df)
        st.subheader("Classificação")
        st.success(f"Tipo de implicação: **{tipo}**")