import streamlit as st from sympy import symbols, Implies, Not, And, Or, truth_table from sympy.logic.boolalg import is_tautology, is_cnf, is_dnf from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.naive_bayes import MultinomialNB import numpy as np import re

st.set_page_config(page_title="Inferências Lógicas com NL + ML", page_icon="🔧") st.title("🔧 Inferências Lógicas com NL + ML") st.markdown("Digite sua condicional em português (Se P, então Q):")

Função para extrair proposições atômicas

def extrair_proposicoes(frase): frase = frase.lower() padrao = r"se (.+?), ent[aã]o (.+)" correspondencia = re.match(padrao, frase) if correspondencia: return correspondencia.group(1).strip(), correspondencia.group(2).strip() return None, None

Função para gerar expressão simbólica

def gerar_expressao(p, q): p_sym, q_sym = symbols('p q') return Implies(p_sym, q_sym), p_sym, q_sym

Função para gerar tabela verdade

def gerar_tabela_verdade(expr, atoms): tabela = [] for i, val in enumerate(truth_table(expr, atoms)): entrada, saida = val.args linha = list(entrada) + [int(saida)] tabela.append(linha) return tabela

Função para prever com ML (simples exemplo com Naive Bayes)

corpus = [ "se chover, então a rua fica molhada", "se estudar, então passa na prova", "se estiver com febre, então está doente", "se p, então q" ] tipos = ["tautologia", "tautologia", "contingência", "tautologia"] vectorizer = TfidfVectorizer() X = vectorizer.fit_transform(corpus) y = np.array(tipos) clf = MultinomialNB().fit(X, y)

def prever_tipo(frase): X_test = vectorizer.transform([frase]) return clf.predict(X_test)[0]

Função para interpretação natural

def gerar_interpretacao(atoms, tipo): p, q = atoms interp = f"{p} não implica que {q} (tipo: {tipo})." if tipo == "tautologia": interp = f"Sempre que {p}, então {q} — ou seja, uma {tipo}." elif tipo == "contradição": interp = f"Nunca ocorre de {p} implicar {q} — isso é uma {tipo}." return interp

Entrada do usuário

frase = st.text_input("Digite aqui:")

if frase: p_texto, q_texto = extrair_proposicoes(frase)

if not p_texto or not q_texto:
    st.error("Frase inválida. Use o formato: 'Se P, então Q'.")
else:
    st.markdown("### 📌 Proposições Atômicas")
    st.write(f"**p:** {p_texto}")
    st.write(f"**q:** {q_texto}")

    expr, p_sym, q_sym = gerar_expressao(p_texto, q_texto)
    st.markdown("### ⚙️ Expressão Simbólica")
    st.code(f"Implies(p, q)")

    tipo_ml = prever_tipo(frase)
    st.markdown(f"**[ML] Tipo previsto:** {tipo_ml.capitalize()}")

    tabela = gerar_tabela_verdade(expr, [p_sym, q_sym])

    contra_exemplo = next((linha for linha in tabela if linha[-1] == 0), None)
    if contra_exemplo:
        st.markdown(f"🚨 **Contra‑exemplo (linha {tabela.index(contra_exemplo)+1}):** p={contra_exemplo[0]}, q={contra_exemplo[1]} ⇒ **Resultado**=0")

    st.markdown("### 🧮 Tabela‑Verdade")
    st.table([["p", "q", "Implies(p, q)"]] + tabela)

    tipo = "tautologia" if all(linha[-1] == 1 for linha in tabela) else (
            "contradição" if all(linha[-1] == 0 for linha in tabela) else "contingência")

    final_tipo = tipo
    if tipo_ml != tipo:
        st.warning("ML divergiu; confira a tabela acima.")
        final_tipo = tipo  # sempre adotar a lógica exata como referência

    st.markdown(f"**[Lógica Exata]**: {final_tipo.capitalize()}")

    st.markdown("### ❌ **Interpretação Natural:**")
    interpretacao = gerar_interpretacao((p_texto, q_texto), final_tipo)
    st.markdown(f"\n> {interpretacao}")

    st.markdown("### 📜 Histórico de Análises")
    st.markdown("""
       
 👤 **Autores:** Valtecir Aragão // Matheus Barbosa // Pedro Favato // Iago Xavier       🎓 **Faculdade:** CEFET-RJ – Sistemas de Informação – Lógica Computacional   

 🔗 [LinkedIn](https://www.linkedin.com/in/valteciraragao)
    """)

