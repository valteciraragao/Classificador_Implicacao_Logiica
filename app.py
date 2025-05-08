import streamlit as st from sympy import symbols, Implies from sympy.logic.inference import is_tautology from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.naive_bayes import MultinomialNB import numpy as np import re

---- MODELO DE MACHINE LEARNING SIMPLES ----

exemplos = [ "Se chover, então a rua fica molhada", "Se Pedro é feliz, então Pedro está feliz", "Se estudar, então passa", "Se trabalhar duro, então terá sucesso", "Se a lâmpada estiver queimada, então a sala estará escura" ]

rotulos = [ "Tautologia", "Tautologia", "Tautologia", "Tautologia", "Contingência" ]

vectorizer = TfidfVectorizer() X = vectorizer.fit_transform(exemplos) y = np.array(rotulos) modelo = MultinomialNB() modelo.fit(X, y)

---- STREAMLIT UI ----

st.title("🔧 Inferências Lógicas com NL + ML") entrada_natural = st.text_input("Digite sua condicional em português (Se P, então Q):")

if entrada_natural: padrao = r"[Ss]e (.+), então (.+)" match = re.match(padrao, entrada_natural)

if match:
    p_texto = match.group(1).strip().lower()
    q_texto = match.group(2).strip().lower()

    p, q = symbols('p q')
    expressao = Implies(p, q)

    st.markdown("### 📌 Proposições Atômicas")
    st.write(f"**p:** {p_texto}")
    st.write(f"**q:** {q_texto}")

    st.markdown("### ⚙️ Expressão Simbólica")
    st.code(str(expressao))

    # ---- PREDIÇÃO ML ----
    entrada_vet = vectorizer.transform([entrada_natural])
    predicao = modelo.predict(entrada_vet)[0]
    st.markdown(f"**[ML] Tipo previsto:** {predicao}")

    # ---- TABELA-VERDADE ----
    st.markdown("### 🧮 Tabela‑Verdade")
    verdades = [(False, False), (False, True), (True, False), (True, True)]
    tabela = []
    for (pv, qv) in verdades:
        resultado = Implies(pv, qv)
        tabela.append((int(pv), int(qv), int(resultado)))
    # Exibe tabela
    df = [dict(p=row[0], q=row[1], **{"Implies(p,q)": row[2]}) for row in tabela]
    st.table(df)

    # ---- LÓGICA EXATA ----
    valor_taut = is_tautology(expressao)
    tipo_exato = "Tautologia" if valor_taut else ("Contradição" if all(r==0 for (_,_,r) in tabela) else "Contingência")
    st.markdown(f"**[Lógica Exata]**: {tipo_exato}")

    if tipo_exato != predicao:
        st.warning("⚠️ ML divergiu; confira a tabela acima.")

    # ---- INTERPRETAÇÃO ----
    st.markdown("### Interpretação Natural")
    if tipo_exato == "Tautologia":
        st.write(f"A implicação entre **{p_texto}** e **{q_texto}** é sempre verdadeira.")
    elif tipo_exato == "Contradição":
        st.write(f"A implicação entre **{p_texto}** e **{q_texto}** é sempre falsa.")
    else:
        st.write(f"**{p_texto}** *não* implica que **{q_texto}** (tipo: *contingência*).")

    # ---- RODAPÉ ----
    st.markdown("""

📜 Histórico de Análises

> 👤 Autores: Valtecir Aragão // Matheus Barbosa // Pedro Favato // Iago Xavier
🎓 Faculdade: CEFET-RJ – Sistemas de Informação – Lógica Computacional
🔗 LinkedIn
""") else: st.error("❌ Erro: Expressão inválida. Use o formato: 'Se P, então Q'")



