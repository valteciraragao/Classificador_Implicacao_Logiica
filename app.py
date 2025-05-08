import streamlit as st from sympy import symbols, Implies from sympy.logic.inference import is_tautology from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.naive_bayes import MultinomialNB import numpy as np import pandas as pd import re

---- MODELO DE MACHINE LEARNING SIMPLES ----

exemplos = [ "Se chover, então a rua fica molhada", "Se Pedro é feliz, então Pedro está feliz", "Se estudar, então passa", "Se trabalhar duro, então terá sucesso", "Se a lâmpada estiver queimada, então a sala estará escura" ] rotulos = ["Tautologia", "Tautologia", "Tautologia", "Tautologia", "Contingência"] vectorizer = TfidfVectorizer() X = vectorizer.fit_transform(exemplos) y = np.array(rotulos) modelo = MultinomialNB() modelo.fit(X, y)

---- STREAMLIT UI ----

st.set_page_config(page_title="Inferências Lógicas com NL + ML") st.title("🔧 Inferências Lógicas com NL + ML")

entrada_natural = st.text_input("Digite sua condicional em português (Se P, então Q):")

if entrada_natural: # Regex para P e Q padrao = r"^[Ss]e\s+(.+),\s+ent[aã]o\s+(.+)$" match = re.match(padrao, entrada_natural)

if match:
    p_texto = match.group(1).strip().lower()
    q_texto = match.group(2).strip().lower()

    # Cria símbolos
    p_sym, q_sym = symbols('p q')
    expressao = Implies(p_sym, q_sym)

    # Exibe átomos
    st.subheader("📌 Proposições Atômicas")
    st.write(f"**p:** {p_texto}")
    st.write(f"**q:** {q_texto}")

    # Exibe expressão simbólica
    st.subheader("⚙️ Expressão Simbólica")
    st.code(str(expressao))

    # Predição ML
    entrada_vet = vectorizer.transform([entrada_natural])
    predicao = modelo.predict(entrada_vet)[0]
    st.markdown(f"**[ML] Tipo previsto:** {predicao}")

    # Gera tabela-verdade
    verdades = [(False, False), (False, True), (True, False), (True, True)]
    linhas = []
    for pv, qv in verdades:
        resultado = Implies(pv, qv)
        linhas.append({"p": int(pv), "q": int(qv), "Implies(p,q)": int(resultado)})
    df_tabela = pd.DataFrame(linhas)
    st.subheader("🧮 Tabela‑Verdade")
    st.table(df_tabela)

    # Lógica exata
    valor_taut = is_tautology(expressao)
    tipo_exato = (
        "Tautologia" if valor_taut else
        "Contradição" if all(r["Implies(p,q)"] == 0 for r in linhas) else
        "Contingência"
    )
    st.markdown(f"**[Lógica Exata]**: {tipo_exato}")
    if tipo_exato != predicao:
        st.warning("⚠️ ML divergiu; confira a tabela acima.")

    # Interpretação natural
    st.subheader("📝 Interpretação Natural")
    if tipo_exato == "Tautologia":
        st.write(f"A implicação entre **{p_texto}** e **{q_texto}** é sempre verdadeira.")
    elif tipo_exato == "Contradição":
        st.write(f"A implicação entre **{p_texto}** e **{q_texto}** é sempre falsa.")
    else:
        st.write(f"**{p_texto}** *não* implica que **{q_texto}** (tipo: *contingência*).")

    # Histórico e rodapé
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Frase": entrada_natural,
        "ML": predicao,
        "Exato": tipo_exato
    })
    df_hist = pd.DataFrame(st.session_state.history)
    st.subheader("📜 Histórico de Análises")
    st.dataframe(df_hist)
else:
    st.error("❌ Erro: Expressão inválida. Use o formato: 'Se P, então Q'")

